#include "post/masking.hpp"
#include "core/encoder_state.hpp"
#include "core/tables.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>

namespace rmps::post {

namespace {

// --- Bark / frequency geometry ---

// Approximate Bark scale (Traunmüller 1990).
double bark_of_hz(double f_hz) {
    return 26.81 * f_hz / (1960.0 + f_hz) - 0.53;
}

// --- Level calibration (MDCT-fullscale based) -----------------------------
//
// The masking model and the ATH need to share a single power<->dB-SPL map so
// that the level-dependent spreading (which needs the masker's SPL) and the
// absolute threshold of hearing (defined in dB SPL) are mutually consistent.
//
// `e_band` here is sum |xr[k]|^2 over the band, in 32-bit-integer MDCT units.
// A loud band sits around 1e14 in these units; we anchor that to ~90 dB SPL,
// giving the affine map below. The offset is the one tuning knob; everything
// downstream (ATH floor, level-dependent slope) follows from it coherently.
constexpr double SPL_OFFSET_DB = -50.0;     // spl ≈ 10·log10(E) + OFFSET

double power_to_spl(double e_band) {
    return 10.0 * std::log10(std::max(e_band, 1.0)) + SPL_OFFSET_DB;
}
double spl_to_power(double spl_db) {
    return std::pow(10.0, (spl_db - SPL_OFFSET_DB) / 10.0);
}

// --- Absolute threshold of hearing (Terhardt), dB SPL ---------------------
//
// ATH(f) ≈ 3.64·(f/1000)^-0.8 - 6.5·exp(-0.6·(f/1000-3.3)²) + 1e-3·(f/1000)^4.
// The quartic term makes the curve climb steeply above ~16 kHz; clamp it so
// it stays finite for the highest bands.
double ath_db_at(double f_hz) {
    const double k = f_hz / 1000.0;
    if (k <= 0.0) return 120.0;
    const double db = 3.64 * std::pow(k, -0.8)
                    - 6.5  * std::exp(-0.6 * (k - 3.3) * (k - 3.3))
                    + 1e-3 * std::pow(k, 4.0);
    return std::clamp(db, -20.0, 120.0);
}

struct SfbGeometry {
    std::array<double, NUM_SFB_LONG_> center_hz{};
    std::array<double, NUM_SFB_LONG_> center_bark{};
    std::array<double, NUM_SFB_LONG_> width_bark{};
    std::array<int,    NUM_SFB_LONG_> n_bins{};
    /// Minimum ATH (dB SPL) over the band. The most audible frequency in the
    /// band — not the band centre — governs whether quantization noise in
    /// the band is audible, so we sample across the band and keep the min.
    std::array<double, NUM_SFB_LONG_> ath_min_db{};
};

SfbGeometry compute_geometry(int samplerate_index, int samplerate) {
    SfbGeometry g{};
    const auto& sfb = tables::scale_fact_band_index[samplerate_index];
    for (int b = 0; b < NUM_SFB_LONG_; ++b) {
        const double lo_sample = sfb[b];
        const double hi_sample = sfb[b + 1];
        const double lo_hz = lo_sample * samplerate / (2.0 * 576.0);
        const double hi_hz = hi_sample * samplerate / (2.0 * 576.0);
        const double c_hz = 0.5 * (lo_hz + hi_hz);
        g.center_hz[b]    = c_hz;
        g.center_bark[b]  = bark_of_hz(c_hz);
        g.width_bark[b]   = std::max(bark_of_hz(hi_hz) - bark_of_hz(lo_hz), 0.05);
        g.n_bins[b]       = static_cast<int>(hi_sample - lo_sample);

        // Sample ATH across the band and keep the minimum (most sensitive).
        constexpr int N_SAMPLES = 8;
        double ath_min = 1e9;
        for (int i = 0; i <= N_SAMPLES; ++i) {
            const double f = lo_hz + (hi_hz - lo_hz) * (i / double(N_SAMPLES));
            ath_min = std::min(ath_min, ath_db_at(std::max(f, 1.0)));
        }
        g.ath_min_db[b] = ath_min;
    }
    return g;
}

// Cache geometry per samplerate index — it never changes.
const SfbGeometry& geometry_for(int samplerate_index, int samplerate) {
    static std::array<SfbGeometry, tables::NUM_SAMPLERATES> cache{};
    static std::array<bool, tables::NUM_SAMPLERATES> ready{};
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    auto idx = static_cast<std::size_t>(samplerate_index);
    if (!ready[idx]) {
        cache[idx] = compute_geometry(samplerate_index, samplerate);
        ready[idx] = true;
    }
    return cache[idx];
}

// --- Spreading function (frequency axis), level-dependent -----------------
//
// Asymmetric Bark-domain spread: a masker at bark z_m raises the threshold at
// z by spread(dz) dB, dz = z - z_m. The downward skirt (toward LF) is steep
// and roughly level-independent (~27 dB/Bark). The upward skirt (toward HF)
// is the one that real loudness research shows to be *level-dependent*: it
// shallows out at high masker SPL, so a loud band spreads much further up the
// spectrum than a quiet one.
//
// Upper slope magnitude (dB/Bark): 24 + 0.23/f_kHz - 0.2·L  (classic MPEG /
// Bosi formulation), clamped to a sane range. Larger L (louder masker) →
// smaller slope → more upward spread. This replaces the old post-hoc
// `level_dep_scale` fudge with a physically motivated mechanism in the spread
// itself.
double spread_db(double dz_bark, double f_masker_khz, double masker_spl_db) {
    if (dz_bark <= 0.0) {
        return 27.0 * dz_bark;                       // downward (dz<0 → negative dB)
    }
    double slope_up = 24.0 + 0.23 / std::max(f_masker_khz, 0.1)
                          - 0.2 * masker_spl_db;
    slope_up = std::clamp(slope_up, 6.0, 30.0);      // dB/Bark, magnitude
    return -slope_up * dz_bark;                       // upward (attenuating)
}

// --- Tonality from Spectral Flatness Measure ---
//
// SFM = exp( mean(log(P_k)) ) / mean(P_k). 1.0 = perfectly flat (noise),
// 0.0 = single tone. We map SFM → tonality coefficient via
// alpha = min(1, -log10(SFM) / 6) clipped, matching Johnston's mapping.
double tonality_from_sfm(const std::int32_t* xrabs, int n_bins) {
    if (n_bins < 2) return 0.5;
    double sum_log = 0.0;
    double sum_lin = 0.0;
    int   counted  = 0;
    for (int k = 0; k < n_bins; ++k) {
        const double v = static_cast<double>(xrabs[k]);
        const double p = v * v + 1.0;  // floor to keep log finite
        sum_log += std::log(p);
        sum_lin += p;
        ++counted;
    }
    const double gm = std::exp(sum_log / counted);
    const double am = sum_lin / counted;
    const double sfm = (am > 0.0) ? std::min(gm / am, 1.0) : 1.0;
    const double dB = -10.0 * std::log10(std::max(sfm, 1e-6));
    // 0 dB (flat) → tonality 0; 60 dB → tonality 1.
    return std::clamp(dB / 60.0, 0.0, 1.0);
}

// --- Per-band SMR from tonality ---
//
// Tone-masking-noise threshold (TMN) used by ISO is roughly 14.5 + b·k dB
// below the masker; noise-masking-tone (NMT) is ~5.5 dB. Interpolate
// linearly between the two on the tonality axis.
double smr_db(double tonality_alpha) {
    constexpr double SMR_NOISE_DB = -5.5;   // noise masker → less margin needed
    constexpr double SMR_TONE_DB  = -14.5;  // tone masker  → more margin (quieter mask)
    return SMR_NOISE_DB + tonality_alpha * (SMR_TONE_DB - SMR_NOISE_DB);
}

// --- Inter-frame tonality from a predictability ("chaos") measure ---------
//
// Single-frame SFM cannot tell a steady tone from band-limited noise that
// happens to be peaky in one frame. The ISO Psy-2 model instead measures how
// *predictable* each spectral line is from the recent past: a stable tone is
// highly predictable (low chaos → tonal), noise is not (high chaos → noisy).
//
// We only have real MDCT magnitudes (no phase), so we use a magnitude-only
// predictor: extrapolate the current magnitude linearly from the two previous
// granules, pred = 2·m1 - m2, and measure the relative error
//     c[k] = |mag - pred| / (mag + |pred|)   ∈ [0, 1].
// Per SFB we take the energy-weighted mean of c[k] and map it to a tonality
// index (ISO): tb = -0.299 - 0.43·ln(c), clamped to [0, 1]. Small c (very
// predictable) → tb near 1 (tonal).
double predictability_tonality(const double* mag, const double* m1,
                               const double* m2, int lo, int hi,
                               bool have_m2) {
    double num = 0.0;   // Σ p·c
    double den = 0.0;   // Σ p
    for (int k = lo; k < hi; ++k) {
        const double a = mag[k];
        double pred;
        if (have_m2) pred = 2.0 * m1[k] - m2[k];   // linear extrapolation
        else         pred = m1[k];                  // hold last (only 1 past)
        if (pred < 0.0) pred = 0.0;
        const double c = std::abs(a - pred) / (a + pred + 1.0);
        const double p = a * a + 1.0;               // energy weight
        num += p * c;
        den += p;
    }
    const double cw = (den > 0.0) ? (num / den) : 1.0;
    const double tb = -0.299 - 0.43 * std::log(std::max(cw, 1e-6));
    return std::clamp(tb, 0.0, 1.0);
}

// --- Low-frequency strictness ---------------------------------------------
//
// Returns a multiplier (≤ 1) applied to the masking threshold so the encoder
// demands lower noise (more bits) at low frequencies. Full `-strict_db` below
// the cutoff, tapering linearly to no-op over ~1 Bark above it to avoid a hard
// band-edge step.
double lf_strict_factor(double center_hz, double cutoff_hz, double strict_db) {
    if (strict_db <= 0.0) return 1.0;
    const double z     = bark_of_hz(center_hz);
    const double z_cut = bark_of_hz(cutoff_hz);
    double t;                                   // 1 = full effect, 0 = none
    if      (z <= z_cut)           t = 1.0;
    else if (z >= z_cut + 1.0)     t = 0.0;
    else                           t = 1.0 - (z - z_cut);   // linear over 1 Bark
    return std::pow(10.0, (-strict_db * t) / 10.0);
}

} // namespace

MaskingParams make_masking_params(const EncoderState& s) {
    MaskingParams p{};
    // λ_R: caller override (>=0) wins; otherwise scale with operating
    // point. At 128 kbps λ_R ≈ 0 (no rate pressure), at 64 kbps it kicks
    // in noticeably. Tune empirically against listening tests.
    if (s.cfg_mask_lambda_R >= 0.0) {
        p.lambda_R = s.cfg_mask_lambda_R;
    } else {
        const double br = static_cast<double>(s.mpeg.bitr);
        p.lambda_R = std::max(0.0, (128.0 - br) / 128.0) * 1.0e9;
    }
    p.alpha = std::clamp(s.cfg_mask_alpha, 0.0, 1.0);
    p.lf_strict_db        = std::max(0.0, s.cfg_lf_strict_db);
    p.lf_strict_cutoff_hz = s.cfg_lf_strict_cutoff_hz;
    return p;
}

void compute_masking_thresholds(const EncoderState& s, int ch, int gr,
                                const MaskingParams& params,
                                MaskingHistory& history,
                                MaskingThresholds& out) {
    const auto& band_table = tables::scale_fact_band_index[s.mpeg.samplerate_index];
    const auto& geom = geometry_for(s.mpeg.samplerate_index, s.wave.samplerate);
    const auto* xr   = s.mdct_freq[ch][gr].data();

    // 1. Per-SFB band energy + tonality.
    std::array<double, NUM_SFB_LONG_> e_band{};
    std::array<double, NUM_SFB_LONG_> tonality{};
    // Per-bin magnitude (double) — needed both for the SFM tonality and as
    // the input to / next state of the inter-frame predictability measure.
    std::array<std::int32_t, GRANULE_SAMPLES> xrabs{};
    std::array<double,       GRANULE_SAMPLES> mag{};
    for (int k = 0; k < GRANULE_SAMPLES; ++k) {
        xrabs[k] = std::abs(xr[k]);
        mag[k]   = static_cast<double>(xrabs[k]);
    }
    for (int b = 0; b < NUM_SFB_LONG_; ++b) {
        const int lo = band_table[b];
        const int hi = band_table[b + 1];
        double sp = 0.0;
        for (int k = lo; k < hi; ++k) {
            const double v = static_cast<double>(xr[k]);
            sp += v * v;
        }
        e_band[b] = sp;

        const double sfm_t = tonality_from_sfm(xrabs.data() + lo, hi - lo);
        if (history.has_hist1) {
            // Blend single-frame SFM with the inter-frame predictability
            // estimate; the latter is the more reliable discriminator.
            const double pred_t = predictability_tonality(
                mag.data(), history.mag_m1.data(), history.mag_m2.data(),
                lo, hi, history.has_hist2);
            const double w = params.inter_frame_tonality_weight;
            tonality[b] = std::clamp(w * pred_t + (1.0 - w) * sfm_t, 0.0, 1.0);
        } else {
            tonality[b] = sfm_t;   // no history yet — SFM alone
        }
    }

    // 2. Apply per-band SMR offset to get the *masker power*.
    //    masker_power[b] = e_band[b] * 10^(SMR/10)
    //    SMR is negative dB (threshold is BELOW masker), so this lowers.
    std::array<double, NUM_SFB_LONG_> masker_power{};
    std::array<double, NUM_SFB_LONG_> masker_spl{};
    for (int b = 0; b < NUM_SFB_LONG_; ++b) {
        const double smr = smr_db(tonality[b]);
        masker_power[b] = e_band[b] * std::pow(10.0, smr / 10.0);
        masker_spl[b]   = power_to_spl(e_band[b]);   // for level-dep slope
    }

    // 3. Spread across the Bark domain (frequency axis): for each target band
    //    b, sum contributions from every masker band m, with a level- and
    //    frequency-dependent upward skirt. Normalize by the total spreading
    //    gain Σ_m g(b,m) so the threshold is a weighted average of masker
    //    powers rather than a raw sum — this removes the geometric bias that
    //    otherwise inflates centre bands (which simply have more neighbours)
    //    relative to edge bands ("double counting").
    std::array<double, NUM_SFB_LONG_> thr{};
    for (int b = 0; b < NUM_SFB_LONG_; ++b) {
        double t    = 0.0;
        double norm = 0.0;
        for (int m = 0; m < NUM_SFB_LONG_; ++m) {
            const double dz   = geom.center_bark[b] - geom.center_bark[m];
            const double s_db = spread_db(dz, geom.center_hz[m] / 1000.0,
                                          masker_spl[m]);
            const double g    = std::pow(10.0, s_db / 10.0);
            t    += masker_power[m] * g;
            norm += g;
        }
        thr[b] = (norm > 0.0) ? (t / norm) : t;
    }

    // 4. Absolute threshold of hearing floor. Uses the band's *minimum* ATH
    //    (its most audible frequency) converted to MDCT power via the shared
    //    SPL calibration, so it is consistent with the masker levels above.
    for (int b = 0; b < NUM_SFB_LONG_; ++b) {
        thr[b] = std::max(thr[b], spl_to_power(geom.ath_min_db[b]));
    }

    // 4b. Low-frequency strictness. Applied AFTER the ATH floor so the bass
    //     threshold may be pushed below absolute audibility — by request, LF
    //     precision is prioritised over bits. Tapers off ~1 Bark above the
    //     cutoff.
    if (params.lf_strict_db > 0.0) {
        for (int b = 0; b < NUM_SFB_LONG_; ++b) {
            thr[b] *= lf_strict_factor(geom.center_hz[b],
                                       params.lf_strict_cutoff_hz,
                                       params.lf_strict_db);
        }
    }

    // 5. Time axis — post-masking: carry forward previous granule's
    //    threshold, decayed. max() = whichever floor is higher wins.
    if (history.has_prev) {
        for (int b = 0; b < NUM_SFB_LONG_; ++b) {
            const double carried = history.prev_thr[b] * params.post_mask_decay;
            if (carried > thr[b]) thr[b] = carried;
        }
    }

    // 6. Time axis — pre-masking penalty for transient/attack-from-silence.
    //    When the granule contains a sub-window of silence preceding an
    //    attack, the band-average threshold overestimates how much masking
    //    actually exists at the silent portion of the granule. Multiply
    //    down so the encoder spends more bits there.
    double pre_mask_mul = 1.0;
    if (gr >= 0 && gr < MAX_GRANULES) {
        if (s.attack_from_silence_flags[gr])
            pre_mask_mul = params.pre_mask_attack_from_silence;
        else if (s.transient_flags[gr] || s.pre_window_flags[gr])
            pre_mask_mul = params.pre_mask_transient;
    }
    if (pre_mask_mul < 1.0) {
        for (int b = 0; b < NUM_SFB_LONG_; ++b) {
            thr[b] *= pre_mask_mul;
        }
    }

    // Persist for next granule's post-mask and predictability measure.
    history.prev_thr = thr;
    history.has_prev = true;
    // Advance the per-bin magnitude window (t-1 → t-2, current → t-1).
    history.mag_m2  = history.mag_m1;
    history.mag_m1  = mag;
    history.has_hist2 = history.has_hist1;   // m2 valid once we had an m1
    history.has_hist1 = true;

    out.thr      = thr;
    out.e_band   = e_band;
    out.tonality = tonality;
}

} // namespace rmps::post
