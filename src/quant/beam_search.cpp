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
#include <cstdio>
#include <unordered_map>
#include <vector>

namespace rmps::quant {

namespace {

constexpr int    BEAM_WIDTH        = 8;
constexpr int    GAIN_WINDOW       = 1;
constexpr int    SF_MAX_CANDIDATE  = 3;
constexpr int    POLISH_ITERS      = 3;
constexpr double NOISE_SHAPING_A   = 0.5;

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

// ---- Per-band shaping quantizer (1st-order error feedback) ----
struct QuantOut {
    bool ok = false;
    std::vector<int> ix;
    int    max_val = 0;
    std::int64_t scalei_band = 0;
    double dequant_factor    = 0.0;
};

QuantOut quantize_band(const std::int32_t* xrabs_band, int n,
                       std::int64_t scalei_global, double steptab_global,
                       int sf, int scale, int preflag, int sfb,
                       const std::array<int, 10000>& int2idx,
                       double alpha) {
    QuantOut out;
    const int eff_sf = sf + (preflag ? PRETAB[static_cast<std::size_t>(sfb)] : 0);
    // The MP3 decoder attenuates each scalefactor band by
    //     2^(-0.5 * (1 + scalefac_scale) * (scalefac + preflag*pretab)).
    // The old code used an integer bit-shift of eff_sf*(1+scale), i.e. TWICE
    // that exponent, so every band with a non-zero scalefactor decoded ~2x too
    // loud — gross broadband distortion that the encoder's (self-consistent)
    // internal SDR could not see. Use the exact decoder factor via pow().
    const double sf_mult = std::pow(2.0, 0.5 * (1 + scale) * eff_sf);
    const std::int64_t scalei = static_cast<std::int64_t>(
        static_cast<double>(scalei_global) * sf_mult + 0.5);
    if (scalei <= 0 || scalei > 0x7fffffffLL) return out;

    const double steptab_band   = steptab_global * sf_mult;
    const double dequant_factor = 4294967296.0 / static_cast<double>(scalei);

    out.ix.resize(static_cast<std::size_t>(n));
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
        if (ln >= 0 && ln < 10000) {
            v = int2idx[static_cast<std::size_t>(ln)];
        } else {
            const double dbl = static_cast<double>(target_i) * steptab_band * 4.656612875e-10;
            v = static_cast<int>(std::pow(std::max(dbl, 0.0), 0.75));
        }
        // Clamp to MP3 max coefficient instead of failing the whole
        // band. The noise shaper above can push a bin over 8191 even
        // when the raw quantizer reports no overflow — failing here
        // empties the cache for this SF and forces an outer_loop
        // fallback later, audible as random LF discontinuity ("boff").
        if (v > 8191) v = 8191;
        out.ix[static_cast<std::size_t>(j)] = v;
        if (v > max_val) max_val = v;

        if (alpha > 0.0) {
            const double x_hat = (v > 0) ? std::pow(static_cast<double>(v), 4.0 / 3.0) * dequant_factor : 0.0;
            err = x_hat - target_f;
        }
    }
    out.ok = true;
    out.max_val = max_val;
    out.scalei_band = scalei;
    out.dequant_factor = dequant_factor;
    return out;
}

double band_distortion(const std::int32_t* xr_band, const std::vector<int>& ix,
                       double dequant_factor, double* out_e_recon = nullptr) {
    double d = 0.0;
    double e_recon = 0.0;
    for (std::size_t j = 0; j < ix.size(); ++j) {
        const int v = ix[j];
        double x_hat_mag = (v > 0) ? std::pow(static_cast<double>(v), 4.0 / 3.0) * dequant_factor : 0.0;
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

int bit_lower_bound(const std::vector<int>& ix) {
    int lb = 0;
    const std::size_t n = ix.size();
    for (std::size_t i = 0; i < n; i += 2) {
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
    subdivide(tmp, s);
    bigv_tab_select(ix_full, tmp);
    bits += bigv_bitcount(ix_full, tmp);
    return bits;
}

// Cache key (sfb, sf) -> (ix, dist, bit_lb).
//   dist  = the hybrid cost the beam ranks on (NMR + SDR + λ_R·bits).
//   D_raw = the raw per-band quantization noise power (Σ (xr - x̂)²);
//           kept for stats so raw_sdr_db / perc_sdr_db remain meaningful
//           regardless of the cost mix.
struct CacheEntry { bool ok = false; std::vector<int> ix; double dist = 0.0;
                    double D_raw = 0.0; double E_recon = 0.0; int bit_lb = 0; };

/// Threshold for "noise-like" content in the upper HF SFBs. Below this
/// tonality we switch the cost from phase-matching to RMS-level-matching.
constexpr double TOP_SFB_NOISE_TONALITY_MAX = 0.30;

/// First SFB that participates in noise-fill mode. At 44.1k this is
/// sfb 17 ≈ 7.5 kHz. The user's perceptual report drove this: with the
/// previous setting (sfb 20 only, 13 kHz+), the 7–13 kHz region got
/// zeroed by the L2 discount + hf_rdo combo and the HF "snap" went away.
constexpr int TOP_SFB_NOISE_FIRST = NUM_SFB_LONG - 4;

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
                               double hf_boost) {
    const bool top_sfb_noise =
        (sfb >= TOP_SFB_NOISE_FIRST) && (tonality_b < TOP_SFB_NOISE_TONALITY_MAX);
    // Apply the HF weight boost to noise-fill SFBs (17–20). Multiplies
    // the perceptual axis only — bit cost is untouched so the search
    // can still trade bits for HF preservation.
    const double weight_eff =
        (sfb >= TOP_SFB_NOISE_FIRST) ? (weight_b * hf_boost) : weight_b;

    double perceptual;
    if (top_sfb_noise) {
        // RMS-level matching with active noise injection up to the
        // masking threshold:
        //   target_e = max(E_band, Thr)
        //   cost = (target_e − E_recon)² / target_e · weight
        //
        // When the source has HF content (E_band > Thr) we track it.
        // When the source is HF-quiet but the threshold is high (e.g.
        // upward spread from a loud kick body covers the HF band), we
        // *inject* quant noise up to the threshold rather than zero the
        // band. Brightness is preserved by spending the smallest possible
        // number of bits on coefficients whose value the listener can't
        // tell apart from the masked noise floor.
        const double target_e = std::max(e_band_b, thr_b);
        const double level_diff = target_e - E_recon_b;
        perceptual = (level_diff * level_diff) / std::max(target_e, 1.0)
                   * weight_eff;
    } else {
        // L1: true hinge — below the masking threshold the noise cost is
        // exactly zero, so the only force on the beam is the bit term.
        const double over = D_b / std::max(thr_b, 1.0);
        const double nmr_cost = (over > 1.0)
            ? std::log(over) * e_band_b * weight_eff
            : 0.0;
        const double sdr_cost = D_b * weight_eff;

        // L2: tonality discount on the perceptual axis only.
        // Floor lifted from 0.2 to 0.4 to preserve HF noise content.
        // On transient granules we disable the discount entirely
        // (factor = 1.0): hi-hat-style HF impulses are exactly the
        // bands that need full bit allocation, and discounting them
        // moved the search's bits to LF where the kick body lives.
        const double tonality_factor = granule_is_transient
            ? 1.0
            : (0.4 + 0.6 * std::clamp(tonality_b, 0.0, 1.0));

        perceptual = (alpha * nmr_cost + (1.0 - alpha) * sdr_cost)
                   * tonality_factor;
    }
    const double bit_cost = lambda_R * static_cast<double>(bit_lb);
    return perceptual + bit_cost;
}

struct BeamState {
    std::vector<std::int8_t> sf;   // tuple of sf per stage
    double cum_dist = 0.0;
    int    cum_bit_lb = 0;
};

void materialize_ix(const std::vector<std::int8_t>& sf_path,
                    const std::vector<std::vector<CacheEntry>>& cache,
                    const std::array<int, tables::NUM_SCFB_LONG>& sfb_table,
                    int* ix_out) {
    std::fill(ix_out, ix_out + GRANULE_SAMPLES, 0);
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const int sf = sf_path[static_cast<std::size_t>(sfb)];
        const auto& e = cache[static_cast<std::size_t>(sfb)][static_cast<std::size_t>(sf)];
        const int start = sfb_table[sfb];
        for (std::size_t j = 0; j < e.ix.size(); ++j) {
            ix_out[start + static_cast<int>(j)] = e.ix[j];
        }
    }
}

} // namespace

BeamResult beam_search(int max_bits, int* ix, int gr, int ch, EncoderState& s,
                       const int* gr0_sf) {
    BeamResult out {};
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
    const bool granule_is_transient =
        s.transient_flags[gr] || s.attack_from_silence_flags[gr]
        || s.pre_window_flags[gr];

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
        std::vector<std::int8_t> sf;
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

        for (int scale : SCALE_CANDIDATES) {
            for (int preflag : PREFLAG_CANDIDATES) {
                for (int gain : viable_gains) {
                    const std::int64_t scalei_global = s.steptabi[gain + 127];
                    const double       steptab_global= s.steptab[gain + 127];

                    // Build (sfb, sf) cache once per (compress, scale, preflag, gain).
                    std::vector<std::vector<CacheEntry>> cache(NUM_SFB_LONG);
                    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
                        const int cap = sf_caps[sfb];
                        cache[static_cast<std::size_t>(sfb)].resize(static_cast<std::size_t>(cap + 1));
                        for (int sf = 0; sf <= cap; ++sf) {
                            auto q = quantize_band(
                                xrabs_bands[sfb], band_lens[sfb],
                                scalei_global, steptab_global,
                                sf, scale, preflag, sfb, s.int2idx,
                                NOISE_SHAPING_A);
                            if (!q.ok) continue;

                            // Active noise injection for noise-fill SFBs.
                            // Whenever the natural quantization zeroes the
                            // band, write a sparse ix=1 pattern so the
                            // decoder synthesises noise targeting 30% of
                            // max(E_band, Thr). Triggers both for "source
                            // had content but quant gain too coarse" and
                            // "source was quiet but threshold has room".
                            const bool noise_fill_band =
                                sfb >= TOP_SFB_NOISE_FIRST
                                && mt.tonality[sfb] < TOP_SFB_NOISE_TONALITY_MAX;
                            if (noise_fill_band && q.max_val == 0) {
                                const double target_e =
                                    std::max(mt.thr[sfb], mt.e_band[sfb]) * 0.30;
                                const double per_bin =
                                    q.dequant_factor * q.dequant_factor;
                                // Only inject when a single ix=1 stays at
                                // or below target_e — overshoot is
                                // penalised more harshly than undershoot
                                // by the level-matching cost, so an
                                // overshooting candidate loses to the
                                // all-zero candidate in the beam.
                                if (per_bin > 0
                                    && per_bin <= target_e
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
                                mt.thr[sfb], mt.e_band[sfb],
                                mt.tonality[sfb],
                                weights[sfb], e.bit_lb,
                                m_alpha, m_lambda_R,
                                granule_is_transient,
                                s.hf_weight_boost);
                            cache[static_cast<std::size_t>(sfb)][static_cast<std::size_t>(sf)] = std::move(e);
                        }
                    }

                    for (std::size_t pi = 0; pi < scfsi_count; ++pi) {
                        const auto& scfsi_pat =
                            (gr == 1 && gr0_sf != nullptr && mpeg1)
                            ? SCFSI_PATTERNS[pi] : SCFSI_PATTERNS[0];

                        const int part2 = part2_length_for(compress, scfsi_pat, gr);
                        const int huff_budget = max_bits - part2;
                        if (huff_budget < 0) continue;

                        std::vector<BeamState> beam(1);
                        beam[0].sf.reserve(NUM_SFB_LONG);
                        bool feasible = true;

                        for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
                            const int sb = scfsi_band_of_sfb(sfb);
                            int forced_sf = -1;
                            if (gr == 1 && gr0_sf != nullptr && scfsi_pat[sb] == 1)
                                forced_sf = gr0_sf[sfb];

                            const int cap = sf_caps[sfb];
                            std::vector<BeamState> next_beam;
                            next_beam.reserve(beam.size() * (forced_sf >= 0 ? 1u : (cap + 1)));

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
                                    ns.sf.push_back(static_cast<std::int8_t>(sf_try));
                                    ns.cum_dist = st.cum_dist + entry.dist;
                                    ns.cum_bit_lb = new_lb;
                                    next_beam.push_back(std::move(ns));
                                };
                                if (forced_sf >= 0) {
                                    if (forced_sf >= 0
                                        && forced_sf < static_cast<int>(cache[static_cast<std::size_t>(sfb)].size())
                                        && cache[static_cast<std::size_t>(sfb)][static_cast<std::size_t>(forced_sf)].ok)
                                        try_sf(forced_sf);
                                } else {
                                    for (int sf_try = 0; sf_try <= cap; ++sf_try) try_sf(sf_try);
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
                            if (found && st.cum_dist >= best.cum_dist) continue;
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
                                best.bits = bits;
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

        // Gain index is steptab[gain+127] with the table sized [0,127], so the
        // valid gain range is [-127, 0]; larger gain ⇒ smaller scalei ⇒
        // coarser ⇒ fewer bits, with gain=0 quantizing ~everything to zero
        // (the guaranteed-feasible extreme). Escalate from the bin-search seed
        // up to 0 and take the *finest* gain that fits. NOTE: gains are
        // negative, so feasibility is tracked with an explicit flag, not a
        // negative sentinel.
        int  chosen_gain = 0;
        bool fb_found    = false;
        std::array<int, GRANULE_SAMPLES> fb_ix {};
        GrInfo fb_gi {};
        int fb_bits = 0;
        const int gstart = std::min(0, std::max(seed_gain, -127));
        for (int g = gstart; g <= 0; ++g) {
            const std::int64_t scalei_g = s.steptabi[g + 127];
            const double       steptab_g = s.steptab[g + 127];
            std::array<int, GRANULE_SAMPLES> t {};
            std::fill(t.begin(), t.end(), 0);
            bool band_ok = true;
            for (int sfb = 0; sfb < NUM_SFB_LONG && band_ok; ++sfb) {
                auto q = quantize_band(xrabs_bands[sfb], band_lens[sfb],
                                       scalei_g, steptab_g, /*sf=*/0, scale,
                                       preflag, sfb, s.int2idx, NOISE_SHAPING_A);
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
        best.sf.assign(NUM_SFB_LONG, 0);
        best.ix = fb_ix;
        best.dist_band.assign(NUM_SFB_LONG, 0.0);
        best.raw_band.assign(NUM_SFB_LONG, 0.0);
        double cum = 0.0;
        for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
            auto q = quantize_band(xrabs_bands[sfb], band_lens[sfb],
                                   scalei_global, steptab_global, /*sf=*/0,
                                   scale, preflag, sfb, s.int2idx, NOISE_SHAPING_A);
            double E_recon = 0.0;
            const double D_raw = q.ok
                ? band_distortion(xr_bands[sfb], q.ix, q.dequant_factor, &E_recon)
                : 0.0;
            const int bit_lb = q.ok ? bit_lower_bound(q.ix) : 0;
            const double dist = hybrid_band_cost(
                sfb, D_raw, E_recon, mt.thr[sfb], mt.e_band[sfb],
                mt.tonality[sfb], weights[sfb], bit_lb,
                m_alpha, m_lambda_R, granule_is_transient, s.hf_weight_boost);
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
                        NOISE_SHAPING_A);
                    if (!q.ok) continue;
                    double new_E_recon = 0.0;
                    const double new_D_raw =
                        band_distortion(xr_bands[sfb], q.ix, q.dequant_factor,
                                        &new_E_recon);
                    const int new_bit_lb = bit_lower_bound(q.ix);
                    const double new_dist = hybrid_band_cost(
                        sfb, new_D_raw, new_E_recon,
                        mt.thr[sfb], mt.e_band[sfb],
                        mt.tonality[sfb],
                        weights[sfb], new_bit_lb,
                        m_alpha, m_lambda_R,
                        granule_is_transient,
                        s.hf_weight_boost);
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
    int hf_end = NUM_SFB_LONG;
    for (int b = TOP_SFB_NOISE_FIRST; b < NUM_SFB_LONG; ++b) {
        if (mt.tonality[b] < TOP_SFB_NOISE_TONALITY_MAX) { hf_end = b; break; }
    }
    const int hf_saved = hf_rdo(best.ix.data(), best.bits,
                                best.dist_band, best.raw_band,
                                max_bits - best.part2_length,
                                best.gain, best.scale, best.preflag,
                                xr_bands, xrabs_bands, band_lens,
                                band_table, weights, s, best.tmp_gi,
                                /*hf_start_sfb=*/11,
                                /*hf_end_sfb=*/hf_end);
    out.hf_rdo_bits_saved = hf_saved;
    if (hf_saved) {
        // best.bits / cum_dist already updated by hf_rdo via reference args.
        best.cum_dist = 0.0;
        for (double d : best.dist_band) best.cum_dist += d;
    }

    // Post-hoc noise injection for HF noise-fill SFBs was attempted
    // here but cannot deliver audible improvement at the bitrates
    // where it would matter: at <128 kbps the slack between huff_budget
    // and best.bits is typically <20 bits — only 1–4 ix=1 injections
    // fit, which is not enough to restore broadband HF noise tonality.
    // Higher bitrate has no need (HF source content already encoded).
    // Left as a known limitation; preserve more HF by raising the ABR
    // target or using --hf-boost > 1.
    if (false) {
    {
        const int huff_budget = max_bits - best.part2_length;
        for (int sfb = TOP_SFB_NOISE_FIRST; sfb < NUM_SFB_LONG; ++sfb) {
            if (mt.tonality[sfb] >= TOP_SFB_NOISE_TONALITY_MAX) continue;
            // Skip if source has essentially no energy in this band.
            if (mt.e_band[sfb] < 1.0) continue;

            const int start = band_table[sfb];
            const int len   = band_lens[sfb];

            // Per-bin reconstruction energy at ix=1 for this granule's
            // (gain, scale, preflag) combo.
            const std::int64_t scalei_global = s.steptabi[best.gain + 127];
            const int eff_sf = 0 + (best.preflag ? tables::pretab[sfb] : 0);
            const int shift  = eff_sf * (1 + best.scale);
            const std::int64_t scalei =
                static_cast<std::int64_t>(scalei_global) << shift;
            if (scalei <= 0 || scalei > 0x7fffffff) continue;
            const double dequant_factor =
                4294967296.0 / static_cast<double>(scalei);
            const double per_bin = dequant_factor * dequant_factor;

            // Current reconstruction energy in this band.
            double cur_e_recon = 0.0;
            int cur_nonzero = 0;
            for (int k = 0; k < len; ++k) {
                const int v = best.ix[start + k];
                if (v > 0) {
                    const double xh = std::pow(static_cast<double>(v), 4.0/3.0)
                                    * dequant_factor;
                    cur_e_recon += xh * xh;
                    ++cur_nonzero;
                }
            }

            // Target = 30% of max(E_band, Thr).
            const double target_e =
                std::max(mt.e_band[sfb], mt.thr[sfb]) * 0.30;

            // Already above target → no injection.
            if (cur_e_recon >= target_e) continue;
            // One additional ix=1 would overshoot — skip rather than blow up.
            if (per_bin > target_e - cur_e_recon) continue;

            // How many additional ix=1 bins we need to reach target_e.
            // Also cap by the actual bit-budget slack: at low bitrate
            // the budget is already mostly consumed, and an injection
            // larger than the slack pushes final_bits past huff_budget
            // (the trial below would revert). Empirically each ix=1
            // in count1 region costs ~2.5 bits including huffman + sign,
            // so use slack/3 as a safe upper bound.
            const int slack = huff_budget - best.bits;
            if (slack < 3) continue;  // not enough slack for even one bin
            // INJECTION_OFF: set to 0 to bypass (A/B baseline measurement).
            const int slack_cap = std::max(1, slack / 3);
            const double deficit = target_e - cur_e_recon;
            int n_add = static_cast<int>(deficit / per_bin);
            const int free_slots = len - cur_nonzero;
            n_add = std::clamp(n_add, 1, std::min(free_slots, slack_cap));
            if (n_add == 0) continue;

            // Tentatively write injection in currently-zero positions.
            std::vector<int> saved(static_cast<std::size_t>(len));
            for (int k = 0; k < len; ++k) saved[k] = best.ix[start + k];
            // Distribute n_add ix=1 across zero slots with uniform stride.
            const int stride_inj = std::max(1, free_slots / n_add);
            int placed = 0;
            int z_idx = 0;
            for (int k = 0; k < len && placed < n_add; ++k) {
                if (best.ix[start + k] != 0) continue;
                if ((z_idx % stride_inj) == (stride_inj / 2)) {
                    best.ix[start + k] = 1;
                    ++placed;
                }
                ++z_idx;
            }
            // Fill remainder if any
            for (int k = 0; k < len && placed < n_add; ++k) {
                if (best.ix[start + k] == 0) {
                    best.ix[start + k] = 1;
                    ++placed;
                }
            }

            // Re-count bits; revert if over budget.
            GrInfo tmp_gi {};
            const int trial_bits = final_bits(best.ix.data(), tmp_gi, s);
            static int inj_dbg = 0;
            if (inj_dbg < 8) {
                std::fprintf(stderr, "[inj#%d] sfb=%d n_add=%d cur_nz=%d trial=%d budget=%d ok=%d\n",
                    inj_dbg, sfb, n_add, cur_nonzero, trial_bits, huff_budget,
                    trial_bits <= huff_budget);
                ++inj_dbg;
            }
            if (trial_bits <= huff_budget) {
                best.tmp_gi = tmp_gi;
                best.bits   = trial_bits;
                // Recompute D_raw / E_recon for this band (other bands
                // are unchanged; we only modified ix in [start, start+len)).
                double new_E_recon = 0.0;
                double new_D = 0.0;
                for (int k = 0; k < len; ++k) {
                    const int v = best.ix[start + k];
                    double xh = (v > 0)
                        ? std::pow(static_cast<double>(v), 4.0/3.0) * dequant_factor
                        : 0.0;
                    new_E_recon += xh * xh;
                    const std::int32_t xr = xr_bands[sfb][k];
                    if (xr < 0) xh = -xh;
                    const double d = static_cast<double>(xr) - xh;
                    new_D += d * d;
                }
                best.raw_band[static_cast<std::size_t>(sfb)] = new_D;
            } else {
                for (int k = 0; k < len; ++k) best.ix[start + k] = saved[k];
            }
        }
    }

    } // end disabled injection block

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
                               /*sfb=*/NUM_SFB_LONG, s.int2idx, NOISE_SHAPING_A);
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
    cod.part2_3_length     = best.part2_length + best.bits;

    out.ok             = true;
    out.part2_3_length = cod.part2_3_length;
    return out;
}

} // namespace rmps::quant
