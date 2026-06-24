#include "quant/beam_search.hpp"
#include "quant/quantize.hpp"
#include "quant/hf_rdo.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include "core/huffman_tables.hpp"
#include "core/tables.hpp"
#include "post/masking.hpp"
#include "post/perceptual.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace rmps::quant {

namespace {

constexpr int    BEAM_WIDTH        = 8;
constexpr int    GAIN_WINDOW       = 1;
constexpr int    SF_MAX_CANDIDATE  = 3;
constexpr int    POLISH_ITERS      = 3;

constexpr std::array<int, 2> COMPRESS_CANDIDATES = {9, 14};
constexpr std::array<int, 2> SCALE_CANDIDATES    = {0, 1};
constexpr std::array<int, 2> PREFLAG_CANDIDATES  = {0, 1};

constexpr std::array<std::array<int, 4>, 4> SCFSI_PATTERNS = {{
    {{0, 0, 0, 0}}, {{1, 1, 0, 0}}, {{0, 0, 1, 1}}, {{1, 1, 1, 1}},
}};

constexpr std::array<int, 5> SCFSI_BOUNDARIES = {0, 6, 11, 16, 21};
constexpr std::array<int, 22> PRETAB =
    {0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,3,3,3,2,0};

int scfsi_band_of_sfb(int sfb) {
    for (int sb = 0; sb < 4; ++sb) {
        if (SCFSI_BOUNDARIES[sb] <= sfb && sfb < SCFSI_BOUNDARIES[sb + 1])
            return sb;
    }
    return 3;
}

int part2_length_for(int compress, const std::array<int, 4>& scfsi_pat, int gr) {
    const int slen1 = tables::slen1_tab[static_cast<std::size_t>(compress)];
    const int slen2 = tables::slen2_tab[static_cast<std::size_t>(compress)];
    int b = 0;
    if (gr == 0 || !scfsi_pat[0]) b += 6 * slen1;
    if (gr == 0 || !scfsi_pat[1]) b += 5 * slen1;
    if (gr == 0 || !scfsi_pat[2]) b += 5 * slen2;
    if (gr == 0 || !scfsi_pat[3]) b += 5 * slen2;
    return b;
}

// ---- Stack-friendly fixed-size array wrapper to avoid heap allocation ----
struct BandIx {
    std::array<int, 256> data{};
    int size = 0;

    BandIx() = default;
    BandIx(const BandIx& other) : size(other.size) {
        if (size > 0) {
            std::copy_n(other.data.data(), size, data.data());
        }
    }
    BandIx& operator=(const BandIx& other) {
        if (this != &other) {
            size = other.size;
            if (size > 0) {
                std::copy_n(other.data.data(), size, data.data());
            }
        }
        return *this;
    }
    BandIx(BandIx&& other) noexcept : size(other.size) {
        if (size > 0) {
            std::copy_n(other.data.data(), size, data.data());
        }
    }
    BandIx& operator=(BandIx&& other) noexcept {
        size = other.size;
        if (size > 0) {
            std::copy_n(other.data.data(), size, data.data());
        }
        return *this;
    }

    int& operator[](std::size_t idx) { return data[idx]; }
    const int& operator[](std::size_t idx) const { return data[idx]; }
};

// ---- Per-band shaping quantizer (1st-order error feedback) ----
struct QuantOut {
    bool ok = false;
    BandIx ix;
    int    max_val = 0;
    std::int64_t scalei_band = 0;
    double dequant_factor    = 0.0;
};

QuantOut quantize_band(const std::int32_t* xrabs_band, int n,
                       std::int64_t scalei_global, [[maybe_unused]] double steptab_global,
                       int sf, int scale, int preflag, int sfb,
                       const std::array<int, 262144>& int2idx,
                       double alpha) {
    QuantOut out;
    const int eff_sf = sf + (preflag ? PRETAB[static_cast<std::size_t>(sfb)] : 0);
    const double sf_mult = tables::pow2_half_tab[static_cast<std::size_t>((1 + scale) * eff_sf)];
    const std::int64_t scalei = static_cast<std::int64_t>(
        static_cast<double>(scalei_global) * sf_mult + 0.5);
    if (scalei <= 0) {
        out.ix.size = n;
        std::fill_n(out.ix.data.begin(), n, 0);
        out.ok = true;
        out.max_val = 0;
        out.scalei_band = 0;
        out.dequant_factor = 0.0;
        return out;
    }
    if (scalei > 0x7fffffffLL) return out;

    const double dequant_factor = 4294967296.0 / static_cast<double>(scalei);

    out.ix.size = n;
    int max_val = 0;
    double err = 0.0;

    for (int j = 0; j < n; ++j) {
        const std::int32_t x = xrabs_band[j];
        double target_f = (alpha > 0.0) ? (static_cast<double>(x) - alpha * err)
                                        : static_cast<double>(x);
        if (target_f < 0.0) target_f = 0.0;
        const std::int32_t target_i = static_cast<std::int32_t>(target_f);

        const std::int64_t prod = static_cast<std::int64_t>(target_i) * scalei
                                + static_cast<std::int64_t>(0x80000000LL);
        std::int32_t ln = fx::i32(prod >> 32);

        int v;
        if (ln >= 0 && ln < 262144) {
            v = int2idx[static_cast<std::size_t>(ln)];
        } else {
            v = 8191;
        }
        if (v > 8191) v = 8191;
        out.ix[static_cast<std::size_t>(j)] = v;
        if (v > max_val) max_val = v;

        if (alpha > 0.0) {
            const double x_hat = tables::pow43_tab[v] * dequant_factor;
            err = x_hat - target_f;
        }
    }
    out.ok = true;
    out.max_val = max_val;
    out.scalei_band = scalei;
    out.dequant_factor = dequant_factor;
    return out;
}

double band_distortion(const std::int32_t* xr_band, const BandIx& ix,
                       double dequant_factor, double* out_e_recon = nullptr) {
    double d = 0.0;
    double e_recon = 0.0;
    for (int j = 0; j < ix.size; ++j) {
        const int v = ix[j];
        double x_hat_mag = tables::pow43_tab[v] * dequant_factor;
        e_recon += x_hat_mag * x_hat_mag;
        double x_hat = x_hat_mag;
        const std::int32_t xr = xr_band[j];
        if (xr < 0) x_hat = -x_hat;
        const double diff = static_cast<double>(xr) - x_hat;
        d += diff * diff;
    }
    if (out_e_recon) *out_e_recon = e_recon;
    return d;
}

int bit_lower_bound(const BandIx& ix) {
    int lb = 0;
    const int n = ix.size;
    for (int i = 0; i < n; i += 2) {
        const int x = ix[i];
        const int y = (i + 1 < n) ? ix[i + 1] : 0;
        if (x != 0 || y != 0) ++lb;
        if (x != 0) ++lb;
        if (y != 0) ++lb;
        if (x > 14) ++lb;
        if (y > 14) ++lb;
    }
    return lb;
}

// Lightweight stand-in passed to bit-count helpers (mirrors loop.py's GrInfoSnapshot).
struct TempGr {
    int big_values = 0;
    int count1 = 0;
    int count1table_select = 0;
    int address1 = 0;
    int address2 = 0;
    int address3 = 0;
    int region0_count = 0;
    int region1_count = 0;
    std::array<int, 3> table_select {0, 0, 0};
};

// We need to call the existing quantize helpers (calc_runlen, count1_bitcount,
// subdivide, bigv_tab_select, bigv_bitcount) which take a `GrInfo&`. Reuse
// them with a real GrInfo we throw away.
int final_bits(int* ix_full, GrInfo& tmp, EncoderState& s) {
    calc_runlen(ix_full, tmp);
    int bits = count1_bitcount(ix_full, tmp);
    // RD-optimal region split (prefix-sum based, cheaper than the legacy
    // fixed-table + heuristic table-select chain). Candidates that only fit
    // the budget under the optimal split become feasible, so the beam can
    // accept finer quantization at the same rate.
    bits += best_subdivide_rd(ix_full, tmp, s);
    return bits;
}

struct CacheEntry {
    bool ok = false;
    BandIx ix;
    double dist = 0.0;
    double D_raw = 0.0;
    double E_recon = 0.0;
    int bit_lb = 0;
};

/// Threshold for "noise-like" content in the upper HF SFBs. Below this
/// tonality we switch the cost from phase-matching to RMS-level-matching.
constexpr double TOP_SFB_NOISE_TONALITY_MAX = 0.30;

/// First SFB that participates in noise-fill mode. At 44.1k sfb 13 ≈ 5 kHz.
/// History: started at sfb 20 (13 kHz+ only), then sfb 17 (~7.5 kHz). Both
/// left the mid-upper band (≈5–11 kHz) to be hard-zeroed by the NMR-hinge /
/// tonality-discount / hf_rdo combination, which showed up as scattered
/// spectral holes ("黒点") in the spectrogram — visibly worse than a
/// reference LAME encode at the same rate. Extending noise-fill down to
/// sfb 13 keeps those noise-like bands continuous (gated by tonality < 0.30,
/// so tonal content is still coded faithfully). Measured segSDR improves at
/// every bitrate (64k +0.2, 96k +0.6, 128k +1.0 dB) with the rate unchanged.
constexpr int TOP_SFB_NOISE_FIRST = NUM_SFB_LONG - 8;

/// Smooth noise-fill weight in [0,1] replacing the old binary
/// `sfb >= FIRST && tonality < 0.30` gate. Two soft ramps:
///   • sfb onset — 0 below (FIRST-RAMP), linearly up to 1 at FIRST.
///   • tonality  — 1 below T_LO (very noise-like), down to 0 at T_HI (tonal).
/// The binary gate flipped a band between faithful coding and noise-fill
/// frame-to-frame whenever its tonality dithered across 0.30; that flip-flop
/// is itself a cause of the spectral-hole flicker ("黒点"). A continuous
/// weight makes near-boundary bands settle on a stable partial fill instead.
constexpr double NF_TON_LO = 0.20;
constexpr double NF_TON_HI = 0.40;
constexpr int    NF_SFB_RAMP = 2;
inline double noise_fill_weight(int sfb, double tonality) {
    double ws;
    if      (sfb >= TOP_SFB_NOISE_FIRST)              ws = 1.0;
    else if (sfb <= TOP_SFB_NOISE_FIRST - NF_SFB_RAMP) ws = 0.0;
    else ws = static_cast<double>(sfb - (TOP_SFB_NOISE_FIRST - NF_SFB_RAMP))
                / static_cast<double>(NF_SFB_RAMP);
    double wt = (NF_TON_HI - tonality) / (NF_TON_HI - NF_TON_LO);
    wt = std::clamp(wt, 0.0, 1.0);
    return ws * wt;
}

/// Hybrid per-band cost:
///     (α · NMR_hinge + (1 - α) · SDR) · tonality_factor + λ_R · bits
///
///  - NMR_hinge = max(0, log(D / Thr)) · E_band · w_b. Sub-threshold
///    quantization noise costs *exactly* zero; the encoder is free to
///    reallocate those bits or to leave them on the floor. This is the
///    "quantization noise is not the enemy" principle.
///  - SDR = D · w_b: legacy A-weighted distortion. Safety net.
///  - tonality_factor in [0.2, 1.0]: noise-like bands get a big discount
///    because quant noise added to a noisy source is perceptually closer
///    to "free" (the listener can't distinguish source noise from quant
///    noise). HF bands are mostly tonality<0.2 so this is where the
///    bit-savings come from. Tonal bands (tonality≈1) get no discount —
///    even sub-threshold pitch shifts on tones are audible.
///  - λ_R · bits: RDO term, NOT discounted by tonality (we want bit
///    savings everywhere, not just in noisy bands).
///
/// Special case: the **topmost SFB** (sfb 20, 13–16 kHz at 44.1k) with
/// low tonality switches to **RMS-level matching**: the cost compares
/// reconstructed energy to source energy rather than per-bin phase. The
/// search is then free to pick a coarse quantization that yields noise
/// at the right level, which is perceptually indistinguishable from the
/// source's own noisy HF content. This is the "noise filling via the
/// existing quantizer" trick — no decoder changes required.
inline double hybrid_band_cost(int sfb, double D_b, double E_recon_b,
                               double thr_b, double e_band_b,
                               double tonality_b,
                               double weight_b, int bit_lb,
                               double alpha, double lambda_R,
                               bool granule_is_transient,
                               double hf_boost,
                               bool is_pre_window = false,
                               double energy_contrast = 1.0,
                               bool is_below_200 = false,
                               double waveform_reg = 0.0) {
    if (is_below_200) {
        // Force minimization of quantization error (distortion D_b) at all costs.
        // We use a massive scale factor so that any distortion in these bands dominates the search,
        // and we ignore bit costs and masking thresholds entirely for these bands.
        return D_b * 1e15;
    }
    // Continuous noise-fill weight (replaces the old binary
    // `sfb>=FIRST && tonality<0.30` switch). w_nf ≈ 1 for noise-like HF
    // bands, ≈ 0 for tonal / low bands, with smooth ramps on both axes.
    // Pre-window granules: suppress noise-fill entirely. Filling the quiet
    // pre-attack region with noise defeats the purpose of pre-echo control.
    const double w_nf = (granule_is_transient || is_pre_window)
                            ? 0.0
                            : noise_fill_weight(sfb, tonality_b);
    // Apply the HF weight boost to noise-fill SFBs. Multiplies the perceptual
    // axis only — bit cost is untouched so the search can still trade bits for
    // HF preservation. Graded by the sfb ramp so there is no hard step.
    const double boost_w = (sfb >= TOP_SFB_NOISE_FIRST) ? 1.0
        : (sfb <= TOP_SFB_NOISE_FIRST - NF_SFB_RAMP) ? 0.0
        : static_cast<double>(sfb - (TOP_SFB_NOISE_FIRST - NF_SFB_RAMP)) / NF_SFB_RAMP;
    double weight_eff = weight_b * (1.0 + boost_w * (hf_boost - 1.0));

    // Pre-window HF penalty: boost the perceptual weight for HF SFBs so
    // the beam search keeps the HF cleaner in the pre-echo zone. The boost
    // is graded by the energy contrast (sharper attack → more exposed
    // pre-echo → stronger penalty). SFB 11+ gets the boost, ramping up.
    if (is_pre_window && sfb >= 11) {
        // Smooth ramp: sfb 11→16 get 2.0x..3.0x, sfb 17→20 get 3.0x..5.0x.
        double pe_boost = 1.0;
        if (sfb >= 17)
            pe_boost = 3.0 + 2.0 * (sfb - 17) / std::max(20 - 17, 1);
        else
            pe_boost = 2.0 + 1.0 * (sfb - 11) / std::max(16 - 11, 1);
        // Scale by energy contrast (diminishing returns via log).
        if (energy_contrast > 1.0)
            pe_boost *= 1.0 + 0.5 * std::log(energy_contrast);
        weight_eff *= pe_boost;
    }

    // Level-matching ("noise-fill") perceptual cost: compare reconstructed
    // band energy to the source/masked target rather than per-bin phase, so
    // the search may pick a coarse quantization that yields noise at the right
    // level — perceptually equivalent to the source's own noisy HF content.
    const double target_e   = std::max(e_band_b, thr_b);
    const double level_diff = target_e - E_recon_b;
    const double p_level = (level_diff * level_diff) / std::max(target_e, 1.0)
                         * weight_eff;

    // Faithful ("phase") perceptual cost: NMR hinge + A-weighted SDR with the
    // tonality discount. Sub-threshold noise costs zero (the hinge).
    const double over = D_b / std::max(thr_b, 1.0);
    const double nmr_cost = (over > 1.0)
        ? std::log(over) * e_band_b * weight_eff
        : 0.0;
    const double sdr_cost = D_b * weight_eff;
    // Pre-window granules: no tonality discount — all noise is exposed.
    const double tonality_factor = (granule_is_transient || is_pre_window)
        ? 1.0
        : (0.4 + 0.6 * std::clamp(tonality_b, 0.0, 1.0));
    const double p_phase = (alpha * nmr_cost + (1.0 - alpha) * sdr_cost)
                         * tonality_factor;

    // Smoothly blend the two regimes by the noise-fill weight. A band whose
    // tonality drifts near the boundary now slides between regimes instead of
    // snapping, removing the frame-to-frame cost-mode flip that flickered the
    // HF as scattered holes.
    double perceptual = (1.0 - w_nf) * p_phase + w_nf * p_level;

    if (sfb >= 11 && E_recon_b == 0.0 && e_band_b > 0.0) {
        // Penalty for zeroing out high-frequency bands when they contain energy.
        // It consists of a base perceptual penalty (active at high bitrates) and
        // a rate-distortion penalty (active at lower bitrates, scaling with lambda_R).
        // Using weight_eff ensures the penalty aligns with the band's audibility/weight.
        constexpr double HF_ZERO_PENALTY_BASE = 2.0e7;
        constexpr double HF_ZERO_PENALTY_BITS = 8.0;
        
        // Boost penalty for noise-like bands to preserve high-frequency noise.
        // tonality_b ranges from 0.0 (noise-like) to 1.0 (tonal).
        const double noise_likeness = 1.0 - std::clamp(tonality_b, 0.0, 1.0);
        // Up to 4.0x penalty multiplier for pure noise bands.
        const double noise_boost = 1.0 + 3.0 * noise_likeness;
        
        perceptual += (HF_ZERO_PENALTY_BASE + HF_ZERO_PENALTY_BITS * lambda_R) * weight_eff * noise_boost;
    }

    if (waveform_reg > 0.0) {
        perceptual += waveform_reg * D_b * weight_eff;
    }

    const double bit_cost = lambda_R * static_cast<double>(bit_lb);
    return perceptual + bit_cost;
}

struct BeamState {
    std::array<std::int8_t, NUM_SFB_LONG> sf {};   // tuple of sf per stage
    int    sf_len = 0;
    double cum_dist = 0.0;
    int    cum_bit_lb = 0;
};

void materialize_ix(const std::array<std::int8_t, NUM_SFB_LONG>& sf_path,
                    const std::array<std::array<CacheEntry, SF_MAX_CANDIDATE + 1>, NUM_SFB_LONG>& cache,
                    const std::array<int, tables::NUM_SCFB_LONG>& sfb_table,
                    int* ix_out) {
    std::fill(ix_out, ix_out + GRANULE_SAMPLES, 0);
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const int sf = sf_path[static_cast<std::size_t>(sfb)];
        const auto& e = cache[static_cast<std::size_t>(sfb)][static_cast<std::size_t>(sf)];
        const int start = sfb_table[sfb];
        for (int j = 0; j < e.ix.size; ++j) {
            ix_out[start + j] = e.ix[j];
        }
    }
}

} // namespace

BeamResult beam_search(int max_bits, int* ix, int gr, int ch, EncoderState& s,
                       const int* gr0_sf) {
    BeamResult out {};
    const double nsa = s.cfg_noise_shaping_alpha;  // noise-shaping coeff (--shaping)
    std::vector<BeamState> beam;
    std::vector<BeamState> next_beam;
    beam.reserve(BEAM_WIDTH);
    next_beam.reserve(BEAM_WIDTH * (SF_MAX_CANDIDATE + 1));

    std::array<std::array<CacheEntry, SF_MAX_CANDIDATE + 1>, NUM_SFB_LONG> cache{};
    auto& cod = s.side_info.gr[gr].ch[ch].tt;
    const auto& band_table = tables::scale_fact_band_index[s.mpeg.samplerate_index];

    const bool is_side = (s.wave.channels == 2 && ch == 1
                          && (s.mpeg.mode_ext & 2));
    const auto& weights = post::perceptual_weights(
        s.mpeg.samplerate_index, s.wave.samplerate, is_side);

    // 3D masking thresholds for this (ch, gr) — precomputed by
    // iteration_loop. Empty (zeros) if the model was bypassed; the
    // hybrid cost then degenerates gracefully to legacy SDR via α=0.
    const auto& mt        = s.mask_thresholds[ch][gr];
    const double m_alpha    = s.mask_params.alpha;
    const double m_lambda_R = s.mask_params.lambda_R;
    const double m_waveform_reg = s.mask_params.waveform_reg;
    const bool granule_is_transient =
        s.transient_flags[gr] || s.attack_from_silence_flags[gr]
        || s.pre_window_flags[gr];
    const bool is_pre_window = s.pre_window_flags[gr]
                            || (gr == 0 && s.cross_frame_pre_window);
    const double gr_energy_contrast = s.energy_contrast[gr];

    // Adapt hf_boost based on bitrate: increase the boost at lower bitrates 
    // to force preservation of high frequencies when bits are scarce.
    double eff_hf_boost = s.hf_weight_boost;
    if (s.mpeg.bitr > 0) {
        double br_factor = 1.0;
        if (s.mpeg.bitr >= 128) {
            double t = static_cast<double>(s.mpeg.bitr - 128) / (320.0 - 128.0);
            br_factor = 1.0 - 0.4 * t; // scales down to 0.6x at 320 kbps
        } else {
            double t = static_cast<double>(128 - s.mpeg.bitr) / (128.0 - 32.0);
            t = std::clamp(t, 0.0, 1.0);
            // Quadratic scaling to boost low bitrates (e.g. 32 kbps, 64 kbps) 
            // more aggressively, up to 3.0x at 32 kbps (was 1.8x linear).
            br_factor = 1.0 + 2.0 * t * t;
        }
        eff_hf_boost = std::max(1.0, s.hf_weight_boost * br_factor);
    }

    // Local mutable copies of weights and thresholds for fundamental and harmonic adjustments
    auto adj_weights = weights;
    auto adj_thr = mt.thr; // Already includes harmonic threshold boosts from the masking model

    int fund_bin = -1;
    std::int32_t max_val = 0;
    // Scan bins 4 to 120 (roughly 150 Hz to 4.5 kHz at 44.1k)
    for (int i = 4; i < 120; ++i) {
        if (s.xrabs[i] > max_val) {
            max_val = s.xrabs[i];
            fund_bin = i;
        }
    }

    if (fund_bin >= 0 && max_val > 1000) {
        auto sfb_containing = [&](int bin) {
            for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
                if (bin >= band_table[sfb] && bin < band_table[sfb + 1]) {
                    return sfb;
                }
            }
            return -1;
        };

        for (int k = 2; k <= 8; ++k) {
            int harm_bin = k * fund_bin;
            if (harm_bin >= GRANULE_SAMPLES) break;
            int sfb = sfb_containing(harm_bin);
            if (sfb >= 0) {
                // Frequencies around harmonics are harder to hear: lower their perceptual weights
                adj_weights[sfb] *= 0.75;
                if (sfb > 0) adj_weights[sfb - 1] *= 0.85;
                if (sfb + 1 < NUM_SFB_LONG) adj_weights[sfb + 1] *= 0.85;
            }
        }
    }

    // Pre-extract band slices.
    std::array<const std::int32_t*, NUM_SFB_LONG> xr_bands {};
    std::array<const std::int32_t*, NUM_SFB_LONG> xrabs_bands {};
    std::array<int, NUM_SFB_LONG> band_lens {};
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const int start = band_table[sfb];
        xr_bands[sfb]    = s.xr     + start;
        xrabs_bands[sfb] = s.xrabs.data() + start;
        band_lens[sfb]   = band_table[sfb + 1] - start;
    }

    // Signal power (for stats and B&B).
    double raw_signal_pow = 0.0;
    double weighted_signal_pow = 0.0;
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        double sp = 0.0;
        for (int j = 0; j < band_lens[sfb]; ++j) {
            const double v = xr_bands[sfb][j];
            sp += v * v;
        }
        raw_signal_pow += sp;
        weighted_signal_pow += sp * weights[sfb];
    }
    out.raw_signal_pow      = raw_signal_pow;
    out.weighted_signal_pow = weighted_signal_pow;

    // High-frequency region (sfb 21, ~16–20 kHz at 44.1k) signal power. The 21
    // transmitted scalefactor bands stop at band_table[NUM_SFB_LONG], but the
    // region above is still coded (with an implicit scalefactor of 0) by the
    // HF post-pass near the end of this function — accumulate its power so the
    // reported SDR stays honest.
    const int hf_start = band_table[NUM_SFB_LONG];
    const int hf_end   = std::min(band_table[NUM_SFB_LONG + 1], GRANULE_SAMPLES);
    double hf_signal_pow = 0.0;
    for (int i = hf_start; i < hf_end; ++i) {
        const double v = s.xr[i];
        hf_signal_pow += v * v;
    }
    out.raw_signal_pow      += hf_signal_pow;
    out.weighted_signal_pow += hf_signal_pow * weights[NUM_SFB_LONG - 1];

    // Seed gain via legacy bin-search.
    cod.scalefac_compress = COMPRESS_CANDIDATES[0];
    int seed_gain = bin_search_StepSize(max_bits, ix, cod, s);

    // R-D pre-screen window.
    std::vector<int> viable_gains;
    {
        const int budget = max_bits - part2_length_for(COMPRESS_CANDIDATES[0], {0,0,0,0}, gr);
        const int budget_max = (budget < 0) ? max_bits : budget;
        std::array<int, GRANULE_SAMPLES> tmp_ix {};
        for (int dg = -GAIN_WINDOW; dg <= GAIN_WINDOW; ++dg) {
            const int g = seed_gain + dg;
            if (g < -120 || g > 119) continue;
            if (quantize(tmp_ix.data(), g, s) > 8192) continue;
            GrInfo tmp {};
            const int bits = final_bits(tmp_ix.data(), tmp, s);
            if (bits >= 0.4 * budget_max && bits <= 1.6 * budget_max)
                viable_gains.push_back(g);
        }
        if (std::find(viable_gains.begin(), viable_gains.end(), seed_gain) == viable_gains.end()
            && seed_gain >= -120 && seed_gain <= 119)
            viable_gains.push_back(seed_gain);
        if (viable_gains.empty()) viable_gains.push_back(seed_gain);
        std::sort(viable_gains.begin(), viable_gains.end());
        viable_gains.erase(std::unique(viable_gains.begin(), viable_gains.end()),
                           viable_gains.end());
    }

    const bool mpeg1 = (s.mpeg.version == 3);
    const std::size_t scfsi_count =
        (gr == 1 && gr0_sf != nullptr && mpeg1) ? SCFSI_PATTERNS.size() : 1u;

    // Best-so-far global tracker.
    struct Best {
        std::array<std::int8_t, NUM_SFB_LONG> sf {};
        std::array<int, GRANULE_SAMPLES> ix {};
        double cum_dist = 0.0;
        std::vector<double> dist_band;   ///< hybrid cost per band (ranking)
        std::vector<double> raw_band;    ///< raw quantization noise per band
        GrInfo tmp_gi {};
        int    bits = 0;
        int    part2_length = 0;
        int    compress = 0;
        int    scale    = 0;
        int    preflag  = 0;
        std::array<int, 4> scfsi_pat {0, 0, 0, 0};
        int    gain     = 0;
    } best {};
    bool found = false;

    // The per-band cache below — quantize_band + band_distortion + the hybrid
    // perceptual cost + the bit lower bound for every (sfb, sf) — is a pure
    // function of (scale, preflag, gain). It does NOT depend on
    // scalefac_compress, which only selects the per-band sf cap and the part2
    // length. The previous ordering put `compress` outermost and rebuilt the
    // whole (dominant) cache for each compress value. Build it ONCE per
    // (scale, preflag, gain) over the full candidate range [0,SF_MAX_CANDIDATE]
    // and reuse it for every COMPRESS_CANDIDATE; the per-compress cap is still
    // enforced in the beam (try_sf and the forced-sf gate), so the emitted
    // bitstream is bit-identical to the old ordering.
    for (int scale : SCALE_CANDIDATES) {
        for (int preflag : PREFLAG_CANDIDATES) {
            for (int gain : viable_gains) {
                const std::int64_t scalei_global = s.steptabi[gain + 127];
                const double       steptab_global= s.steptab[gain + 127];

                // Reset + build the full cache (sf = 0..SF_MAX_CANDIDATE).
                for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb)
                    for (int sf = 0; sf <= SF_MAX_CANDIDATE; ++sf)
                        cache[sfb][sf].ok = false;
                for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
                    const double lo_hz = band_table[sfb] * s.wave.samplerate / 1152.0;
                    const bool is_below_200 = (lo_hz < 200.0);
                    for (int sf = 0; sf <= SF_MAX_CANDIDATE; ++sf) {
                        auto q = quantize_band(
                            xrabs_bands[sfb], band_lens[sfb],
                            scalei_global, steptab_global,
                            sf, scale, preflag, sfb, s.int2idx,
                            nsa);
                        if (!q.ok) continue;

                        // Active noise injection for noise-fill SFBs.
                        // Whenever the natural quantization zeroes the
                        // band, write a sparse ix=1 pattern so the
                        // decoder synthesises noise targeting 30% of
                        // max(E_band, Thr). Triggers both for "source
                        // had content but quant gain too coarse" and
                        // "source was quiet but threshold has room".
                        const double nf_w = noise_fill_weight(sfb, mt.tonality[sfb]);
                        if (nf_w > 0.0 && q.max_val == 0) {
                            // Seed the hole toward the source's own band level
                            // (envelope-tracking) graded by nf_w, instead of a
                            // flat 30 % — fills holes to a level that follows
                            // the source spectral shape, and tapers to nothing
                            // at the sfb/tonality boundaries (no seam).
                            const double target_frac = (lo_hz >= 12000.0) ? 0.85 : 0.55;
                            const double target_e =
                                std::max(mt.thr[sfb], mt.e_band[sfb]) * (target_frac * nf_w);
                            const double per_bin =
                                q.dequant_factor * q.dequant_factor;
                            // Relax the overshoot gate above 12 kHz to allow PNS-like noise injection 
                            // even under coarser quantization (up to 1.8x overshoot).
                            const bool can_inject = (lo_hz >= 12000.0)
                                ? (per_bin <= target_e * 1.8)
                                : (per_bin <= target_e);
                            if (per_bin > 0
                                && can_inject
                                && target_e > 0) {
                                int n_target =
                                    static_cast<int>(target_e / per_bin);
                                n_target = std::clamp(n_target,
                                                      1, band_lens[sfb]);
                                const int stride_inj = std::max(1,
                                    band_lens[sfb] / n_target);
                                for (int pos = stride_inj / 2;
                                     pos < band_lens[sfb];
                                     pos += stride_inj) {
                                    q.ix[static_cast<std::size_t>(pos)] = 1;
                                }
                            }
                        }

                        CacheEntry e;
                        e.ok = true;
                        e.ix = q.ix;
                        e.D_raw = band_distortion(xr_bands[sfb], q.ix, q.dequant_factor,
                                                 &e.E_recon);
                        e.bit_lb = bit_lower_bound(q.ix);
                        e.dist = hybrid_band_cost(
                            sfb, e.D_raw, e.E_recon,
                            adj_thr[sfb], mt.e_band[sfb],
                            mt.tonality[sfb],
                            adj_weights[sfb], e.bit_lb,
                            m_alpha, m_lambda_R,
                            granule_is_transient,
                            eff_hf_boost,
                            is_pre_window,
                            gr_energy_contrast,
                            is_below_200,
                            m_waveform_reg);
                        cache[static_cast<std::size_t>(sfb)][static_cast<std::size_t>(sf)] = std::move(e);
                    }
                }

                for (int compress : COMPRESS_CANDIDATES) {
                    const int slen1 = tables::slen1_tab[static_cast<std::size_t>(compress)];
                    const int slen2 = tables::slen2_tab[static_cast<std::size_t>(compress)];

                    std::array<int, NUM_SFB_LONG> sf_caps {};
                    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
                        const int slen = (sfb < 11) ? slen1 : slen2;
                        int cap = (slen > 0) ? ((1 << slen) - 1) : 0;
                        cap = std::min(cap, SF_MAX_CANDIDATE);
                        sf_caps[sfb] = cap;
                    }

                    for (std::size_t pi = 0; pi < scfsi_count; ++pi) {
                        const auto& scfsi_pat =
                            (gr == 1 && gr0_sf != nullptr && mpeg1)
                            ? SCFSI_PATTERNS[pi] : SCFSI_PATTERNS[0];

                        const int part2 = part2_length_for(compress, scfsi_pat, gr);
                        const int huff_budget = max_bits - part2;
                        if (huff_budget < 0) continue;

                        beam.clear();
                        beam.resize(1);
                        beam[0].sf.fill(0);
                        beam[0].sf_len = 0;
                        beam[0].cum_dist = 0.0;
                        beam[0].cum_bit_lb = 0;
                        bool feasible = true;

                        for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
                            const double lo_hz = band_table[sfb] * s.wave.samplerate / 1152.0;
                            const bool is_below_200 = (lo_hz < 200.0);
                            const int sb = scfsi_band_of_sfb(sfb);
                            int forced_sf = -1;
                            if (gr == 1 && gr0_sf != nullptr && scfsi_pat[sb] == 1)
                                forced_sf = gr0_sf[sfb];

                            const int cap = sf_caps[sfb];
                            next_beam.clear();

                            for (auto& st : beam) {
                                auto try_sf = [&](int sf_try) {
                                    if (sf_try < 0) return;
                                    if (sf_try > cap && forced_sf < 0) return;
                                    auto& entry =
                                        cache[static_cast<std::size_t>(sfb)]
                                             [static_cast<std::size_t>(sf_try)];
                                    if (!entry.ok) return;
                                    const int new_lb = st.cum_bit_lb + entry.bit_lb;
                                    if (new_lb > huff_budget) return;
                                     BeamState ns;
                                     ns.sf = st.sf;
                                     ns.sf[sfb] = static_cast<std::int8_t>(sf_try);
                                     ns.sf_len = sfb + 1;
                                     ns.cum_dist = st.cum_dist + entry.dist;
                                     ns.cum_bit_lb = new_lb;
                                     next_beam.push_back(ns);
                                };
                                if (forced_sf >= 0) {
                                    // The cache now spans the full
                                    // [0,SF_MAX_CANDIDATE] range irrespective of
                                    // compress, so gate the forced (scfsi-shared)
                                    // sf by this compress's cap to reproduce the
                                    // old per-compress build exactly.
                                    if (forced_sf <= cap
                                        && forced_sf < static_cast<int>(cache[static_cast<std::size_t>(sfb)].size())
                                        && cache[static_cast<std::size_t>(sfb)][static_cast<std::size_t>(forced_sf)].ok)
                                        try_sf(forced_sf);
                                } else {
                                    if (is_below_200) {
                                        try_sf(cap);
                                    } else {
                                        for (int sf_try = 0; sf_try <= cap; ++sf_try) try_sf(sf_try);
                                    }
                                }
                            }
                            if (next_beam.empty()) { feasible = false; break; }

                            std::partial_sort(next_beam.begin(),
                                              next_beam.begin()
                                                + std::min<std::size_t>(BEAM_WIDTH, next_beam.size()),
                                              next_beam.end(),
                                              [](const BeamState& a, const BeamState& b) {
                                                  return a.cum_dist < b.cum_dist;
                                              });
                            if (next_beam.size() > BEAM_WIDTH) next_beam.resize(BEAM_WIDTH);
                            beam = std::move(next_beam);
                        }
                        if (!feasible) continue;

                        // Materialise survivors and run real bit count.
                        std::array<int, GRANULE_SAMPLES> tmp_ix {};
                        for (auto& st : beam) {
                            if (found && st.cum_dist >= best.cum_dist) break;
                            materialize_ix(st.sf, cache, band_table, tmp_ix.data());
                            GrInfo tmp_gi {};
                            const int bits = final_bits(tmp_ix.data(), tmp_gi, s);
                            if (bits > huff_budget) continue;
                            if (!found || st.cum_dist < best.cum_dist) {
                                best.sf = st.sf;
                                best.ix = tmp_ix;
                                best.cum_dist = st.cum_dist;
                                best.dist_band.resize(NUM_SFB_LONG);
                                best.raw_band.resize(NUM_SFB_LONG);
                                for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
                                    const auto& ce = cache[static_cast<std::size_t>(sfb)]
                                                          [static_cast<std::size_t>(st.sf[static_cast<std::size_t>(sfb)])];
                                    best.dist_band[static_cast<std::size_t>(sfb)] = ce.dist;
                                    best.raw_band[static_cast<std::size_t>(sfb)]  = ce.D_raw;
                                }
                                best.tmp_gi = tmp_gi;
                                best.bits   = bits;
                                best.part2_length = part2;
                                best.compress = compress;
                                best.scale    = scale;
                                best.preflag  = preflag;
                                best.scfsi_pat = scfsi_pat;
                                best.gain      = gain;
                                found = true;
                            }
                        }
                    }
                }
            }
        }
    }

    if (!found) {
        // Guaranteed-feasible in-model fallback. The beam keeps the lowest-
        // distortion (= highest-bit) survivors, so when the budget is tight
        // every survivor can overflow huff_budget on the real bit count and
        // the search yields nothing — previously handing off to the legacy
        // outer_loop, whose reset scalefactors cause audible LF
        // discontinuities ("boff"). Instead we escalate the global gain,
        // using the *same* shaping quantizer the trellis uses, until the
        // all-sf=0 encoding actually fits, and build `best` from it. The
        // greedy ±1 polish below then refines the per-band sf allocation
        // against the masking thresholds within budget. At the coarse
        // extreme all coefficients quantize to zero, so this terminates.
        const int compress = COMPRESS_CANDIDATES[0];
        const int scale = 0, preflag = 0;
        const auto& scfsi_pat = SCFSI_PATTERNS[0];
        const int part2 = part2_length_for(compress, scfsi_pat, gr);
        const int huff_budget = std::max(0, max_bits - part2);

        // Gain index is steptab[gain+127] with the table sized [0,255], so the
        // valid gain range is [-127, 127]; larger gain ⇒ smaller scalei ⇒
        // coarser ⇒ fewer bits, with gain=127 quantizing ~everything to zero
        // (the guaranteed-feasible extreme). Escalate from the bin-search seed
        // up to 127 and take the *finest* gain that fits. NOTE: feasibility
        // is tracked with an explicit flag, not a negative sentinel.
        int  chosen_gain = 0;
        bool fb_found    = false;
        std::array<int, GRANULE_SAMPLES> fb_ix {};
        GrInfo fb_gi {};
        int fb_bits = 0;
        const int gstart = std::min(127, std::max(seed_gain, -127));
        for (int g = gstart; g <= 127; ++g) {
            const std::int64_t scalei_g = s.steptabi[g + 127];
            const double       steptab_g = s.steptab[g + 127];
            std::array<int, GRANULE_SAMPLES> t {};
            std::fill(t.begin(), t.end(), 0);
            bool band_ok = true;
            for (int sfb = 0; sfb < NUM_SFB_LONG && band_ok; ++sfb) {
                const double lo_hz = band_table[sfb] * s.wave.samplerate / 1152.0;
                const bool is_below_200 = (lo_hz < 200.0);
                int sf_val = 0;
                if (is_below_200) {
                    const int slen = (sfb < 11)
                        ? tables::slen1_tab[static_cast<std::size_t>(compress)]
                        : tables::slen2_tab[static_cast<std::size_t>(compress)];
                    int cap = (slen > 0) ? ((1 << slen) - 1) : 0;
                    sf_val = std::min(cap, SF_MAX_CANDIDATE);
                }
                auto q = quantize_band(xrabs_bands[sfb], band_lens[sfb],
                                       scalei_g, steptab_g, sf_val, scale,
                                       preflag, sfb, s.int2idx, nsa);
                if (!q.ok) { band_ok = false; break; }
                const int start = band_table[sfb];
                for (int j = 0; j < band_lens[sfb]; ++j)
                    t[static_cast<std::size_t>(start + j)] = q.ix[static_cast<std::size_t>(j)];
            }
            if (!band_ok) continue;
            GrInfo tg {};
            const int bits = final_bits(t.data(), tg, s);
            if (bits <= huff_budget) {
                chosen_gain = g; fb_ix = t; fb_gi = tg; fb_bits = bits;
                fb_found = true; break;
            }
        }
        if (!fb_found) return out;   // pathological: part2 alone > budget

        const std::int64_t scalei_global = s.steptabi[chosen_gain + 127];
        const double       steptab_global = s.steptab[chosen_gain + 127];
        best.sf.fill(0);
        best.ix = fb_ix;
        best.dist_band.assign(NUM_SFB_LONG, 0.0);
        best.raw_band.assign(NUM_SFB_LONG, 0.0);
        double cum = 0.0;
        for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
            const double lo_hz = band_table[sfb] * s.wave.samplerate / 1152.0;
            const bool is_below_200 = (lo_hz < 200.0);
            int sf_val = 0;
            if (is_below_200) {
                const int slen = (sfb < 11)
                    ? tables::slen1_tab[static_cast<std::size_t>(compress)]
                    : tables::slen2_tab[static_cast<std::size_t>(compress)];
                int cap = (slen > 0) ? ((1 << slen) - 1) : 0;
                sf_val = std::min(cap, SF_MAX_CANDIDATE);
            }
            best.sf[static_cast<std::size_t>(sfb)] = static_cast<std::int8_t>(sf_val);

            auto q = quantize_band(xrabs_bands[sfb], band_lens[sfb],
                                   scalei_global, steptab_global, sf_val,
                                   scale, preflag, sfb, s.int2idx, nsa);
            double E_recon = 0.0;
            const double D_raw = q.ok
                ? band_distortion(xr_bands[sfb], q.ix, q.dequant_factor, &E_recon)
                : 0.0;
            const int bit_lb = q.ok ? bit_lower_bound(q.ix) : 0;
            const double dist = hybrid_band_cost(
                sfb, D_raw, E_recon, adj_thr[sfb], mt.e_band[sfb],
                mt.tonality[sfb], adj_weights[sfb], bit_lb,
                m_alpha, m_lambda_R, granule_is_transient, eff_hf_boost,
                is_pre_window, gr_energy_contrast, is_below_200,
                m_waveform_reg);
            best.dist_band[static_cast<std::size_t>(sfb)] = dist;
            best.raw_band[static_cast<std::size_t>(sfb)]  = D_raw;
            cum += dist;
        }
        best.cum_dist     = cum;
        best.tmp_gi       = fb_gi;
        best.bits         = fb_bits;
        best.part2_length = part2;
        best.compress     = compress;
        best.scale        = scale;
        best.preflag      = preflag;
        best.scfsi_pat    = scfsi_pat;
        best.gain         = chosen_gain;
        found = true;
    }

    // --- Local greedy ±1 polish (operates on materialised ix) ---
    {
        auto scalei_global = s.steptabi[best.gain + 127];
        auto steptab_global = s.steptab[best.gain + 127];
        const int huff_budget = max_bits - best.part2_length;
        std::array<bool, NUM_SFB_LONG> locked {};
        if (gr == 1 && gr0_sf != nullptr) {
            for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb)
                locked[sfb] = (best.scfsi_pat[scfsi_band_of_sfb(sfb)] == 1);
        }

        for (int it = 0; it < POLISH_ITERS; ++it) {
            bool improved = false;
            for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
                if (locked[sfb]) continue;
                const double lo_hz = band_table[sfb] * s.wave.samplerate / 1152.0;
                const bool is_below_200 = (lo_hz < 200.0);
                const int slen = (sfb < 11)
                    ? tables::slen1_tab[static_cast<std::size_t>(best.compress)]
                    : tables::slen2_tab[static_cast<std::size_t>(best.compress)];
                int cap = (slen > 0) ? ((1 << slen) - 1) : 0;
                cap = std::min(cap, SF_MAX_CANDIDATE);
                const int cur_sf = best.sf[static_cast<std::size_t>(sfb)];
                for (int dlt : {-1, +1}) {
                    const int new_sf = cur_sf + dlt;
                    if (new_sf < 0 || new_sf > cap) continue;
                    auto q = quantize_band(
                        xrabs_bands[sfb], band_lens[sfb],
                        scalei_global, steptab_global,
                        new_sf, best.scale, best.preflag, sfb, s.int2idx,
                        nsa);
                    if (!q.ok) continue;
                    double new_E_recon = 0.0;
                    const double new_D_raw =
                        band_distortion(xr_bands[sfb], q.ix, q.dequant_factor,
                                        &new_E_recon);
                    const int new_bit_lb = bit_lower_bound(q.ix);
                    const double new_dist = hybrid_band_cost(
                        sfb, new_D_raw, new_E_recon,
                        adj_thr[sfb], mt.e_band[sfb],
                        mt.tonality[sfb],
                        adj_weights[sfb], new_bit_lb,
                        m_alpha, m_lambda_R,
                        granule_is_transient,
                        eff_hf_boost,
                        is_pre_window,
                        gr_energy_contrast,
                        is_below_200,
                        m_waveform_reg);
                    if (new_dist >= best.dist_band[static_cast<std::size_t>(sfb)]) continue;

                    // Apply tentatively.
                    const int start = band_table[sfb];
                    std::vector<int> saved(static_cast<std::size_t>(band_lens[sfb]));
                    for (int j = 0; j < band_lens[sfb]; ++j) {
                        saved[static_cast<std::size_t>(j)] = best.ix[start + j];
                        best.ix[start + j] = q.ix[static_cast<std::size_t>(j)];
                    }
                    GrInfo tmp_gi {};
                    const int trial_bits = final_bits(best.ix.data(), tmp_gi, s);
                    if (trial_bits <= huff_budget) {
                        best.cum_dist += (new_dist - best.dist_band[static_cast<std::size_t>(sfb)]);
                        best.dist_band[static_cast<std::size_t>(sfb)] = new_dist;
                        best.raw_band[static_cast<std::size_t>(sfb)]  = new_D_raw;
                        best.sf[static_cast<std::size_t>(sfb)] = static_cast<std::int8_t>(new_sf);
                        best.tmp_gi = tmp_gi;
                        best.bits   = trial_bits;
                        improved    = true;
                    } else {
                        for (int j = 0; j < band_lens[sfb]; ++j)
                            best.ix[start + j] = saved[static_cast<std::size_t>(j)];
                    }
                }
            }
            if (!improved) break;
        }
    }

    // --- HF rate-distortion polish ---
    // Any SFB in noise-fill mode uses level-matching cost; zeroing those
    // bands via hf_rdo would punish the level match. Stop hf_rdo at the
    // first noise-fill SFB whose tonality is below the threshold.
    int hf_rdo_end_sfb = NUM_SFB_LONG;
    for (int b = TOP_SFB_NOISE_FIRST; b < NUM_SFB_LONG; ++b) {
        if (mt.tonality[b] < TOP_SFB_NOISE_TONALITY_MAX) { hf_rdo_end_sfb = b; break; }
    }
    const int hf_saved = hf_rdo(best.ix.data(), best.bits,
                                best.dist_band, best.raw_band,
                                max_bits - best.part2_length,
                                best.gain, best.scale, best.preflag,
                                xr_bands, xrabs_bands, band_lens,
                                band_table, adj_weights, mt.tonality, s, best.tmp_gi,
                                /*hf_start_sfb=*/11,
                                /*hf_end_sfb=*/hf_rdo_end_sfb);
    out.hf_rdo_bits_saved = hf_saved;
    if (hf_saved) {
        // best.bits / cum_dist already updated by hf_rdo via reference args.
        best.cum_dist = 0.0;
        for (double d : best.dist_band) best.cum_dist += d;
    }

    // --- High-frequency region post-pass (sfb 21, ~16–20 kHz at 44.1k) ---
    // The beam only allocates the 21 transmitted scalefactor bands (up to
    // band_table[NUM_SFB_LONG] ≈ 16 kHz). The region above is legal MP3
    // content coded by the big_values/count1 machinery with an *implicit*
    // scalefactor of 0 — only the global gain applies and nothing extra is
    // sent in the side info. Quantise it at the chosen global gain and keep
    // it whenever it still fits the bit budget, so genuine >16 kHz content
    // (e.g. from hi-res 24/32-bit masters) survives instead of being hard-
    // zeroed. When the band is silent the quantizer yields all zeros and the
    // splice costs nothing; when the budget is tight the candidate is
    // reverted, preserving the previous behaviour at low bitrates.
    double hf_raw_dist = 0.0;
    if (hf_start < hf_end) {
        const std::int64_t scalei_global = s.steptabi[best.gain + 127];
        const double       steptab_global = s.steptab[best.gain + 127];
        auto q = quantize_band(s.xrabs.data() + hf_start, hf_end - hf_start,
                               scalei_global, steptab_global,
                               /*sf=*/0, best.scale, best.preflag,
                               /*sfb=*/NUM_SFB_LONG, s.int2idx, nsa);
        if (q.ok && q.max_val > 0) {
            for (int j = 0; j < hf_end - hf_start; ++j)
                best.ix[static_cast<std::size_t>(hf_start + j)] =
                    q.ix[static_cast<std::size_t>(j)];
            GrInfo hf_gi {};
            const int hf_bits = final_bits(best.ix.data(), hf_gi, s);
            if (hf_bits <= max_bits - best.part2_length) {
                best.tmp_gi = hf_gi;
                best.bits   = hf_bits;
                double e_recon = 0.0;
                hf_raw_dist = band_distortion(s.xr + hf_start, q.ix,
                                              q.dequant_factor, &e_recon);
            } else {
                for (int j = 0; j < hf_end - hf_start; ++j)
                    best.ix[static_cast<std::size_t>(hf_start + j)] = 0;
            }
        }
    }

    // --- Black-spot guard ---
    // A granule whose every coefficient quantises to zero decodes as a silent
    // frame — an audible spectral hole / dropout ("黒点"). Budget starvation
    // and the coarse fallback extreme (gain→127) can both reach this state. If
    // the source granule actually carries energy, force the single most
    // audible coded bin to ix=1: a few count1 bits the reservoir can always
    // absorb, turning a dropout into a faint-but-present granule. This cannot
    // fire on normal granules (they already have non-zero coefficients), so it
    // is inert outside true starvation.
    if (s.xrmax > 0) {
        bool all_zero = true;
        for (int i = 0; i < GRANULE_SAMPLES; ++i)
            if (best.ix[i] != 0) { all_zero = false; break; }
        if (all_zero) {
            int    floor_bin   = -1;
            double floor_score = 0.0;
            for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
                const int    start = band_table[sfb];
                const double w     = weights[static_cast<std::size_t>(sfb)];
                for (int j = 0; j < band_lens[sfb]; ++j) {
                    const double sc = static_cast<double>(xrabs_bands[sfb][j]) * w;
                    if (sc > floor_score) { floor_score = sc; floor_bin = start + j; }
                }
            }
            if (floor_bin >= 0) {
                best.ix[static_cast<std::size_t>(floor_bin)] = 1;
                GrInfo fg {};
                const int fb = final_bits(best.ix.data(), fg, s);
                // Keep the bin unconditionally: a single count1 coefficient is
                // only a few bits, never approaches the 4095-bit hard granule
                // cap, and any overshoot of the (soft) per-granule share flows
                // through the cross-frame reservoir. Reverting here would
                // re-introduce the very dropout this guard exists to remove, so
                // black-spot avoidance wins over the advisory bit target.
                best.tmp_gi = fg;
                best.bits   = fb;
            }
        }
    }

    // --- Compute raw distortion for stats ---
    double raw_dist = 0.0;
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        raw_dist += best.raw_band[static_cast<std::size_t>(sfb)];
    }
    raw_dist += hf_raw_dist;
    out.raw_dist = raw_dist;
    out.cum_dist = best.cum_dist + hf_raw_dist * weights[NUM_SFB_LONG - 1];

    // --- Apply best to cod_info / scalefactor / ix output ---
    cod.scalefac_compress  = best.compress;
    cod.scalefac_scale     = best.scale;
    cod.preflag            = best.preflag;
    cod.quantizerStepSize  = best.gain;
    cod.part2_length       = best.part2_length;
    if (gr == 1 && gr0_sf != nullptr && mpeg1) {
        for (int b = 0; b < 4; ++b) s.side_info.scfsi[ch][b] = best.scfsi_pat[b];
    }
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb)
        s.scalefactor.l[gr][ch][sfb] = best.sf[static_cast<std::size_t>(sfb)];
    for (int i = 0; i < GRANULE_SAMPLES; ++i) ix[i] = best.ix[i];

    cod.big_values         = best.tmp_gi.big_values;
    cod.count1             = best.tmp_gi.count1;
    cod.count1table_select = best.tmp_gi.count1table_select;
    cod.address1           = best.tmp_gi.address1;
    cod.address2           = best.tmp_gi.address2;
    cod.address3           = best.tmp_gi.address3;
    cod.region0_count      = best.tmp_gi.region0_count;
    cod.region1_count      = best.tmp_gi.region1_count;
    cod.table_select       = best.tmp_gi.table_select;

    // --- RD-optimal Huffman region split (finalisation-only) ---
    // The beam ranks candidates with the fast fixed-table subdivide; once the
    // winner is chosen, re-split its big-values region optimally. This is a
    // pure bit reclaim: the recount can only shrink part2_3_length, and every
    // bit saved flows into the cross-frame reservoir.
    {
        const int c1_bits   = count1_bitcount(ix, cod);
        const int bigv_bits = best_subdivide_rd(ix, cod, s);
        cod.part2_3_length  = best.part2_length + c1_bits + bigv_bits;
    }

    out.ok             = true;
    out.part2_3_length = cod.part2_3_length;
    return out;
}

} // namespace rmps::quant
