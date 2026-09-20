#include "post/ms_stereo.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include <cstdlib>
#include "core/tables.hpp"

#include <algorithm>
#include <cmath>

namespace rmps::post {

namespace {
constexpr double INV_SQRT2 = 0.7071067811865475;

} // namespace

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


/// Drop the weaker of M/S in bands where the other clearly dominates.
///
/// Above roughly 1.5 kHz the binaural system stops resolving interaural phase
/// — the same fact `psy::side_threshold_offset_db` already encodes, where the
/// Side threshold offset turns from a 15 dB *penalty* at low frequency into a
/// 3 dB relief above 6 kHz. Where phase is not resolved the image is carried
/// by envelope, so a band whose energy sits almost entirely in one of the two
/// rotated channels loses nothing audible by coding only that one: the other
/// quantizes to a handful of near-zero values that still cost Huffman bits
/// and buy an inter-channel detail the ear cannot use at that frequency.
///
/// Zeroing a band is a *constraint*, not a re-weighting, which is why it can
/// work at all here: this encoder's cost function has repeatedly proved inert
/// to re-priced terms and responsive to anything that removes options.
///
/// Two rules, and they are the same mechanism:
///   * keep only the stronger side when one dominates;
///   * a band that is not dominant in either channel is not worth its bits,
///     but *one* side is always kept — the conditions below are mutually
///     exclusive because the ratio is < 1, so both can never fire on the same
///     band and a band can never be emptied.
///
/// Zeroing S makes the band mono (L = R = M/sqrt2), which is what M/S coding
/// already does whenever S quantizes away. Zeroing M makes it fully
/// out-of-phase, which is only reached when S genuinely dominates — i.e. the
/// source really is anti-correlated there.
void prune_dominated_bands(EncoderState& s,
                           std::array<std::int32_t, GRANULE_SAMPLES>& M,
                           std::array<std::int32_t, GRANULE_SAMPLES>& S) {
    static const double floor_hz = [] {
        if (const char* e = std::getenv("RMPS_MS_PRUNE_HZ")) {
            const double v = std::atof(e);
            if (v >= 0.0) return v;
        }
        // Default off — measured, it does not pay off.
        //
        // The idea is sound and the bit saving is real: in quality VBR,
        // 3 kHz/8 dB took 220 -> 202 kbps (-8.2 %). But a lower bitrate at a
        // fixed `-q` is not efficiency, it is a lower operating point, and
        // rate-matched at CBR 128 the harness makes it a net loss —
        // 6 kHz/8 dB: segSNR_p05 0 better / 2 worse (mixed_music@128 -1.43),
        // hole rate 0 / 1. 6 kHz/20 dB is bit-identical to off.
        //
        // Two measurement traps this walked into, both worth remembering:
        //  * `snr.py` and `fbscore.py` collapse the decode with
        //    `x.mean(axis=1)` — that is the Mid channel, so they are nearly
        //    blind to anything that only touches Side. `stereo.py` reports M,
        //    S and image width separately and is the right instrument here.
        //  * Judging a bit-saving change in quality VBR confounds "freed
        //    bits" with "lowered rate"; at CBR the freed bits have to go
        //    somewhere and the trade becomes visible.
        //
        // What the stereo metric shows: at 128 kbps the Side channel above
        // 2 kHz is already almost entirely gone (error -4.19 dB at 2-6 kHz,
        // -0.01 dB at 11-16 kHz, i.e. error as large as signal) and width is
        // already -6.4/-8.8 dB above 6 kHz from the existing side_cut_fraction
        // lowpass. There is very little left for this to save.
        return 0.0;                     // 0 = disabled
    }();
    if (floor_hz <= 0.0) return;
    static const double ratio_db = [] {
        if (const char* e = std::getenv("RMPS_MS_PRUNE_DB")) {
            const double v = std::atof(e);
            if (v > 0.0) return v;
        }
        return 20.0;                    // weaker must be this far down
    }();
    const double ratio = std::pow(10.0, -ratio_db / 10.0);   // power ratio

    const auto& bt = tables::scale_fact_band_index[s.mpeg.samplerate_index];
    const double bin_hz = static_cast<double>(s.wave.samplerate) / 1152.0;
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        if (bt[sfb] * bin_hz < floor_hz) continue;
        double em = 0.0, es = 0.0;
        for (int i = bt[sfb]; i < bt[sfb + 1]; ++i) {
            const double m = M[i], sd = S[i];
            em += m * m;
            es += sd * sd;
        }
        if (em <= 0.0 && es <= 0.0) continue;
        if (es < em * ratio) {
            for (int i = bt[sfb]; i < bt[sfb + 1]; ++i) S[i] = 0;
        } else if (em < es * ratio) {
            for (int i = bt[sfb]; i < bt[sfb + 1]; ++i) M[i] = 0;
        }
    }
}

} // namespace

void apply_ms_stereo(EncoderState& s) {
    if (s.wave.channels != 2) return;

    // ----- Adaptive joint-stereo mode decision (per frame) -----
    // At low bitrates (<= 48 kbps), lock to M/S stereo mode (mode_ext = 2)
    // to prevent frame-to-frame L/R <-> M/S mode switching flickering, which causes Side audio to sound choppy.
    if (s.cfg_adaptive_ms && s.policy_bitr() > 48) {
        const bool use_ms = frame_prefers_ms(s);
        s.ms_prev_frame   = use_ms;
        s.mpeg.mode_ext   = use_ms ? 2 : 0;
        if (!use_ms) return;   // keep plain L/R — skip the rotation entirely
    } else {
        s.ms_prev_frame = true;
        s.mpeg.mode_ext = 2;
    }

    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
        auto& L = s.mdct_freq[0][gr];
        auto& R = s.mdct_freq[1][gr];

        // The rotation is not bounded by its inputs: M = (L+R)/√2 reaches
        // √2·|L| when the channels agree, so a centred coefficient at 70 % of
        // int32 lands past the top. Storing that through `fx::i32` wrapped it
        // and inverted the sign — the original "boff".
        //
        // Clipping it is not the answer either. A coefficient pinned at
        // INT32_MAX poisons everything downstream that normalises by the
        // granule's peak: `e_ref` in the band cost, the masking thresholds,
        // the prominence weight. Measured, one such granule was granted 468
        // bits and chose to spend **30**, coding essentially nothing, because
        // relative to that one giant value the rest of the spectrum looked
        // inaudible. It reads as a one-granule hole — 6.5 dB down at 1-2 kHz,
        // recovering in the next granule. That is the "poko".
        //
        // So scale instead of clip. The internal MDCT scale is arbitrary per
        // (channel, granule) — `global_gain` is derived from the data and
        // absorbs it — so shrinking the whole granule to fit changes nothing
        // in the decoded output, and it keeps the spectrum's shape exactly.
        double peak = 0.0;
        for (int i = 0; i < GRANULE_SAMPLES; ++i) {
            const double l = L[i], r = R[i];
            peak = std::max(peak, std::abs(l + r));
            peak = std::max(peak, std::abs(l - r));
        }
        peak *= INV_SQRT2;
        // Scale only what does not fit, and only just enough.
        //
        // Do not be tempted to aim lower "for headroom": the internal scale is
        // *not* free. Targeting 2^30 instead of the top of the range was tried
        // and it is a disaster — it fires on every loud granule rather than
        // only on overflowing ones, throwing away a bit of quantizer
        // resolution across the board. Measured: segSNR −5.13 dB mean, 0 cells
        // better and 20 worse; p05 −4.25 dB, 0 better and 20 worse. Audible
        // immediately.
        const double fit = (peak > 2147483647.0) ? (2147483647.0 / peak) : 1.0;
        const double k = INV_SQRT2 * fit;

        for (int i = 0; i < GRANULE_SAMPLES; ++i) {
            const std::int64_t l = L[i], r = R[i];
            L[i] = fx::sat32(static_cast<double>(l + r) * k);
            R[i] = fx::sat32(static_cast<double>(l - r) * k);
        }

        prune_dominated_bands(s, L, R);
    }
}

} // namespace rmps::post
