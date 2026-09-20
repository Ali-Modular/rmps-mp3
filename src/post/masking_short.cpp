#include "post/masking_short.hpp"
#include "post/psy_common.hpp"
#include "core/encoder_state.hpp"
#include "core/tables.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>

namespace rmps::post {

namespace {

// --- Cross-transform level calibration -------------------------------------
//
// The SPL map in psy_common (and therefore the ATH floor) is calibrated
// against *long* MDCT units. The short transform sums 12 windowed input
// samples per output instead of 36, so a steady component lands in a short
// line with one third the amplitude — one ninth the power — of the long line
// it would occupy. Two constants follow, and they are the only place the two
// transforms' scales meet:
//
//   • a per-component quantity (the ATH, which describes one just-audible
//     tone) converts by 1/9;
//   • a per-band total summed over all three windows converts by 1/3, because
//     three windows each hold one ninth: 3 × 1/9 = 1/3.
constexpr double SHORT_LINE_POWER_SCALE = 1.0 / 9.0;
constexpr double SHORT_GRANULE_ENERGY_SCALE = 1.0 / 3.0;

/// Forward masking between adjacent short windows. The long model's decay
/// coefficient describes one full granule (~13 ms); a short window is one
/// third of that, so the per-window coefficient is its cube root. Deriving it
/// this way keeps the two models consistent instead of introducing a second,
/// independently-tuned time constant.
double window_forward_decay(double granule_decay) {
    return std::pow(std::clamp(granule_decay, 1e-6, 1.0), 1.0 / 3.0);
}

/// Backward (pre-) masking from the following window.
///
/// Backward masking is both weak and short — it is essentially gone by 2 ms,
/// and consecutive short windows sit 4.3 ms apart. A small but non-zero value
/// keeps the model from demanding absurd precision in a window that is about
/// to be buried, without letting an attack excuse the noise that precedes it.
constexpr double WINDOW_BACKWARD_MASK = 0.05;   // −13 dB

struct ShortGeometry {
    std::array<double, NUM_SFB_SHORT_PSY> center_hz{};
    std::array<double, NUM_SFB_SHORT_PSY> lo_bark{};
    std::array<double, NUM_SFB_SHORT_PSY> hi_bark{};
    std::array<double, NUM_SFB_SHORT_PSY> width_bark{};
    std::array<int,    NUM_SFB_SHORT_PSY> n_lines{};
    std::array<double, NUM_SFB_SHORT_PSY> ath_pow{};
    std::array<double, NUM_SFB_SHORT_PSY> bark_frac{};
    /// Start of each (sfb, window) run in transmission order.
    std::array<std::array<int, NUM_WINDOWS>, NUM_SFB_SHORT_PSY> seg_begin{};
};

ShortGeometry compute_short_geometry(int samplerate_index, int samplerate) {
    ShortGeometry g{};
    const auto& sb = tables::scale_fact_band_index_short[samplerate_index];
    // A short window holds 192 lines spanning DC..Nyquist.
    const double hz_per_line = samplerate / (2.0 * 192.0);
    int pos = 0;
    for (int b = 0; b < NUM_SFB_SHORT_PSY; ++b) {
        const double lo_hz = sb[b] * hz_per_line;
        const double hi_hz = sb[b + 1] * hz_per_line;
        g.center_hz[b]  = 0.5 * (lo_hz + hi_hz);
        g.lo_bark[b]    = psy::bark_of_hz(lo_hz);
        g.hi_bark[b]    = psy::bark_of_hz(hi_hz);
        g.width_bark[b] = std::max(g.hi_bark[b] - g.lo_bark[b], 0.05);
        g.n_lines[b]    = sb[b + 1] - sb[b];
        g.ath_pow[b]    = psy::spl_to_power(psy::ath_min_db_over(lo_hz, hi_hz))
                        * SHORT_LINE_POWER_SCALE;
        // Transmission order groups (sfb, window) runs contiguously — the same
        // walk build_segments() performs in the short quantizer.
        for (int w = 0; w < NUM_WINDOWS; ++w) {
            g.seg_begin[b][w] = pos;
            pos += g.n_lines[b];
        }
    }
    const double z_lo = g.lo_bark[0];
    const double z_hi = std::max(g.hi_bark[NUM_SFB_SHORT_PSY - 1], z_lo + 1e-6);
    for (int b = 0; b < NUM_SFB_SHORT_PSY; ++b) {
        const double zc = 0.5 * (g.lo_bark[b] + g.hi_bark[b]);
        g.bark_frac[b] = std::clamp((zc - z_lo) / (z_hi - z_lo), 0.0, 1.0);
    }
    return g;
}

const ShortGeometry& short_geometry_for(int samplerate_index, int samplerate) {
    static std::array<ShortGeometry, tables::NUM_SAMPLERATES> cache{};
    static std::array<bool, tables::NUM_SAMPLERATES> ready{};
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    const auto idx = static_cast<std::size_t>(samplerate_index);
    if (!ready[idx]) {
        cache[idx] = compute_short_geometry(samplerate_index, samplerate);
        ready[idx] = true;
    }
    return cache[idx];
}

} // namespace

void compute_short_masking_thresholds(const EncoderState& s, int ch, int gr,
                                      const MaskingParams& params,
                                      ShortMaskingHistory& history,
                                      ShortMaskingThresholds& out) {
    const auto& geom = short_geometry_for(s.mpeg.samplerate_index,
                                          s.wave.samplerate);
    const auto* xr = s.mdct_freq[ch][gr].data();

    // --- 1. Per-(window, sfb) energy and tonality ---
    //
    // Tonality comes from the single-window spectral flatness alone. The long
    // model additionally uses an inter-frame predictability measure, but over
    // 4.3 ms windows during the transient that made the encoder pick short
    // blocks in the first place, that predictor is measuring the transient,
    // not the tone/noise character — flatness is the more trustworthy signal
    // at this resolution.
    std::array<std::array<double, NUM_SFB_SHORT_PSY>, NUM_WINDOWS> e_band{};
    std::array<std::array<double, NUM_SFB_SHORT_PSY>, NUM_WINDOWS> tonality{};
    std::array<std::int32_t, GRANULE_SAMPLES> xrabs{};
    for (int i = 0; i < GRANULE_SAMPLES; ++i) xrabs[i] = std::abs(xr[i]);

    for (int b = 0; b < NUM_SFB_SHORT_PSY; ++b) {
        const int n = geom.n_lines[b];
        for (int w = 0; w < NUM_WINDOWS; ++w) {
            const int beg = geom.seg_begin[b][w];
            double e = 0.0;
            for (int i = beg; i < beg + n; ++i) {
                const double v = static_cast<double>(xr[i]);
                e += v * v;
            }
            e_band[w][b]   = e;
            tonality[w][b] = psy::tonality_from_sfm(xrabs.data() + beg, n);
        }
    }

    // --- 2. Frequency spreading, per window, then ATH and LF strictness ---
    std::array<std::array<double, NUM_SFB_SHORT_PSY>, NUM_WINDOWS> base{};
    for (int w = 0; w < NUM_WINDOWS; ++w) {
        std::array<double, NUM_SFB_SHORT_PSY> spread_e{};
        std::array<double, NUM_SFB_SHORT_PSY> spread_ton_num{};

        for (int m = 0; m < NUM_SFB_SHORT_PSY; ++m) {
            const double e_m = e_band[w][m];
            if (e_m <= 0.0) continue;
            // Convert to long-equivalent power so the level-dependent slope
            // reads the same SPL it would for the same signal on a long block.
            const double spl = psy::effective_masker_spl(
                e_m / SHORT_LINE_POWER_SCALE, geom.width_bark[m],
                tonality[w][m]);
            const double a_u = psy::upper_slope_db_per_bark(
                                   geom.center_hz[m] / 1000.0, spl)
                             * psy::LN10_OVER_10;
            const double a_d = psy::LOWER_SLOPE_DB_PER_BARK * psy::LN10_OVER_10;
            const double C   = psy::spread_norm(a_u, a_d);
            const double w_m = geom.width_bark[m];

            for (int b = 0; b < NUM_SFB_SHORT_PSY; ++b) {
                double S;
                if (b == m) {
                    S = psy::spread_self(C, a_u, a_d, w_m);
                } else if (b > m) {
                    const double gap =
                        std::max(0.0, geom.lo_bark[b] - geom.hi_bark[m]);
                    S = psy::spread_fraction(C, a_u, w_m, geom.width_bark[b], gap);
                } else {
                    const double gap =
                        std::max(0.0, geom.lo_bark[m] - geom.hi_bark[b]);
                    S = psy::spread_fraction(C, a_d, w_m, geom.width_bark[b], gap);
                }
                const double contrib = e_m * S;
                spread_e[b]       += contrib;
                spread_ton_num[b] += contrib * tonality[w][m];
            }
        }

        for (int b = 0; b < NUM_SFB_SHORT_PSY; ++b) {
            const double eff_ton = (spread_e[b] > 0.0)
                ? std::clamp(spread_ton_num[b] / spread_e[b], 0.0, 1.0)
                : tonality[w][b];
            double t = spread_e[b] * std::pow(10.0, psy::smr_db(eff_ton) / 10.0);
            t = std::max(t, geom.ath_pow[b]);
            if (params.lf_strict_db > 0.0)
                t *= psy::lf_strict_factor(geom.center_hz[b],
                                           params.lf_strict_cutoff_hz,
                                           params.lf_strict_db);
            base[w][b] = t;
        }
    }

    // --- 3. Temporal masking across the three windows ---
    //
    // Forward masking chains (window 0 helps 1, which helps 2), so it reads
    // the already-carried threshold of the previous window. Backward masking
    // does not accumulate over 8 ms, so it reads the *base* threshold of the
    // immediately following window only.
    auto& thr = out.thr;
    for (int w = 0; w < NUM_WINDOWS; ++w) {
        for (int b = 0; b < NUM_SFB_SHORT_PSY; ++b) {
            const double granule_decay = params.post_mask_decay_lf
                + geom.bark_frac[b]
                    * (params.post_mask_decay_hf - params.post_mask_decay_lf);
            const double fwd = window_forward_decay(granule_decay);

            double t = base[w][b];

            const double prev = (w > 0)
                ? thr[w - 1][b]
                : (history.has_prev ? history.prev_thr[b] : 0.0);
            if (prev > 0.0) t = std::max(t, prev * fwd);

            if (w + 1 < NUM_WINDOWS)
                t = std::max(t, base[w + 1][b] * WINDOW_BACKWARD_MASK);

            thr[w][b] = t;
        }
    }

    out.e_band = e_band;
    out.valid  = true;

    // --- 4. Perceptual entropy over the whole granule ---
    double pe = 0.0;
    for (int w = 0; w < NUM_WINDOWS; ++w)
        pe += psy::perceptual_entropy_of(e_band[w].data(), thr[w].data(),
                                         geom.n_lines.data(),
                                         NUM_SFB_SHORT_PSY);
    out.pe = pe;

    history.prev_thr = thr[NUM_WINDOWS - 1];
    history.has_prev = true;
    (void)gr;
    (void)ch;
}

void project_short_to_long(const EncoderState& s,
                           const ShortMaskingThresholds& st,
                           MaskingHistory& long_history) {
    if (!st.valid) return;

    const auto& geom_s = short_geometry_for(s.mpeg.samplerate_index,
                                            s.wave.samplerate);
    const auto& long_table =
        tables::scale_fact_band_index[s.mpeg.samplerate_index];
    const double hz_per_short_line = s.wave.samplerate / (2.0 * 192.0);
    const double hz_per_long_line  = s.wave.samplerate / (2.0 * 576.0);

    // Take the granule's *last* window as the state that carries forward — it
    // is the one temporally adjacent to the next granule — and express it as a
    // power density per short line, so it can be re-integrated over whatever
    // width the long bands have.
    std::array<double, NUM_SFB_SHORT_PSY> density{};
    for (int b = 0; b < NUM_SFB_SHORT_PSY; ++b) {
        const int n = std::max(geom_s.n_lines[b], 1);
        density[b] = st.thr[NUM_WINDOWS - 1][b] / n;
    }

    for (int lb = 0; lb < NUM_SFB_LONG_; ++lb) {
        const double lo_hz = long_table[lb] * hz_per_long_line;
        const double hi_hz = long_table[lb + 1] * hz_per_long_line;
        const int n_long   = long_table[lb + 1] - long_table[lb];

        // Average the short density over the long band's frequency span.
        double acc = 0.0;
        double wsum = 0.0;
        for (int sb = 0; sb < NUM_SFB_SHORT_PSY; ++sb) {
            const double s_lo =
                tables::scale_fact_band_index_short[s.mpeg.samplerate_index][sb]
                * hz_per_short_line;
            const double s_hi =
                tables::scale_fact_band_index_short[s.mpeg.samplerate_index][sb + 1]
                * hz_per_short_line;
            const double ov = std::min(hi_hz, s_hi) - std::max(lo_hz, s_lo);
            if (ov <= 0.0) continue;
            acc  += density[sb] * ov;
            wsum += ov;
        }
        if (wsum <= 0.0) continue;

        // density is per *short* line; a long line is one third as wide, so
        // it holds one third of the power at equal PSD — and the long
        // transform scales that back up by the granule energy factor.
        const double per_long_line =
            (acc / wsum) / SHORT_GRANULE_ENERGY_SCALE / 3.0;
        long_history.prev_thr[lb] = per_long_line * n_long;
    }
    long_history.has_prev = true;
}

} // namespace rmps::post
