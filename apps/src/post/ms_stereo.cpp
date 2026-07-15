#include "post/ms_stereo.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include "core/tables.hpp"

#include <algorithm>
#include <cmath>

namespace rmps::post {

namespace { constexpr double INV_SQRT2 = 0.7071067811865475; }

namespace {

/// Per-frame M/S vs L/R decision.
///
/// Proxy for the bit cost of coding a channel pair: per scalefactor band,
/// bits ~ width · log2(1 + E_band / width). The constant masking-threshold
/// factor cancels in the comparison, so the *difference* between the L/R
/// and M/S sums is a faithful relative rate estimate. M/S wins exactly when
/// the rotation concentrates energy (correlated material); on decorrelated
/// or anti-correlated material E[M] ≈ E[S] ≈ E[L] ≈ E[R] and the rotation
/// buys nothing while still forcing the Side channel through the tighter
/// Side LPF/folding path — there L/R is strictly better.
bool frame_prefers_ms(const EncoderState& s) {
    const auto& band_table = tables::scale_fact_band_index[s.mpeg.samplerate_index];
    double cost_lr = 0.0, cost_ms = 0.0;
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
        const auto& L = s.mdct_freq[0][gr];
        const auto& R = s.mdct_freq[1][gr];
        for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
            const int start = band_table[sfb];
            const int end   = band_table[sfb + 1];
            const double w  = static_cast<double>(end - start);
            double el = 0.0, er = 0.0, cr = 0.0;
            for (int i = start; i < end; ++i) {
                const double l = L[i], r = R[i];
                el += l * l;
                er += r * r;
                cr += l * r;
            }
            // m = (l+r)/√2, t = (l−r)/√2  ⇒  E_m/E_s from el, er and the
            // cross term — no second pass over the coefficients needed.
            const double em = std::max(0.0, 0.5 * (el + er) + cr);
            const double es = std::max(0.0, 0.5 * (el + er) - cr);
            cost_lr += w * (std::log2(1.0 + el / w) + std::log2(1.0 + er / w));
            cost_ms += w * (std::log2(1.0 + em / w) + std::log2(1.0 + es / w));
        }
    }
    // ~2 % hysteresis around the previous frame's decision: a stereo-image
    // mode flip is itself mildly audible, so only switch when the rate win
    // is real, not measurement dither.
    const double margin = s.ms_prev_frame ? 1.02 : 0.98;
    return cost_ms <= cost_lr * margin;
}

} // namespace

void apply_ms_stereo(EncoderState& s) {
    if (s.wave.channels != 2) return;

    // ----- Adaptive joint-stereo mode decision (per frame) -----
    if (s.cfg_adaptive_ms) {
        const bool use_ms = frame_prefers_ms(s);
        s.ms_prev_frame   = use_ms;
        s.mpeg.mode_ext   = use_ms ? 2 : 0;
        if (!use_ms) return;   // keep plain L/R — skip the rotation entirely
    } else {
        s.mpeg.mode_ext = 2;
    }

    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
        auto& L = s.mdct_freq[0][gr];
        auto& R = s.mdct_freq[1][gr];
        for (int i = 0; i < GRANULE_SAMPLES; ++i) {
            const std::int64_t l = L[i], r = R[i];
            const double m = static_cast<double>(l + r) * INV_SQRT2;
            const double t = static_cast<double>(l - r) * INV_SQRT2;
            L[i] = fx::i32(static_cast<std::int64_t>(m));
            R[i] = fx::i32(static_cast<std::int64_t>(t));
        }
    }
}

} // namespace rmps::post
