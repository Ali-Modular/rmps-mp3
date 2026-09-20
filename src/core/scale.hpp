#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace rmps::scale {

/// Left shift applied to each int16 PCM sample on entry to the analysis
/// chain (subband.cpp). Everything in the MDCT domain is scaled by 2^shift
/// relative to the sample value.
///
/// This exists as one accessor because the encoder was *not* scale-invariant:
/// several absolute thresholds were written as literals calibrated against
/// `int16 << 16` and had no idea the convention could move. Changing the
/// shift alone therefore did not buy headroom, it broke the psychoacoustic
/// model — 16 -> 15 measured 13.96 -> 6.03 dB SNR, 16 -> 14 -> 2.50 dB, when
/// a pure input-scale change should have been absorbed by `global_gain` and
/// cost nothing at all.
///
/// Every threshold that lives in the MDCT domain now derives its value from
/// here, so the shift is a real knob: reducing it gives the transform
/// headroom (the MDCT is a coherent sum, so a loud tonal bin comes out larger
/// than the sample that produced it and was landing on INT32_MAX — measured
/// at bin 1, the kick fundamental) without moving the model underneath it.
inline int headroom_shift() {
    static const int v = [] {
        if (const char* e = std::getenv("RMPS_HEADROOM_SHIFT")) {
            const int n = std::atoi(e);
            if (n >= 10 && n <= 16) return n;
        }
        // 15 gives the transform one bit of headroom, which is enough: the
        // INT32_MAX saturation on loud bass is gone by then and 14 measures no
        // better (32.99 vs 33.09 dB). Against the old 16 this is +19.0 dB of
        // SNR on real material at 320 kbps — the saturation was the dominant
        // error, not quantization.
        return 15;
    }();
    return v;
}

/// The shift the surviving literals were calibrated at.
inline constexpr int REF_SHIFT = 16;

/// How far the MDCT domain has moved from that calibration, in dB of power.
/// Negative when the shift is reduced (the domain got quieter).
inline double mdct_scale_db() {
    return 6.020599913279624 * (headroom_shift() - REF_SHIFT);
}

/// Linear power factor for the same move. Multiply a reference-calibrated
/// energy threshold by this to keep it at the same acoustic level.
inline double power_factor() {
    const int d = headroom_shift() - REF_SHIFT;
    return std::pow(4.0, d);          // amplitude 2^d, power 4^d
}

/// Amplitude factor. Multiply a reference-calibrated magnitude threshold.
inline double amplitude_factor() {
    return std::pow(2.0, headroom_shift() - REF_SHIFT);
}

/// A magnitude threshold expressed at the reference shift, moved to the
/// current one. Never returns 0 for a positive input, so a floor stays a
/// floor.
inline std::int32_t magnitude(std::int32_t at_ref) {
    const double v = at_ref * amplitude_factor();
    const std::int32_t r = static_cast<std::int32_t>(v + 0.5);
    return (at_ref > 0 && r < 1) ? 1 : r;
}

/// Correction to the `global_gain = stepSize + 210` offset.
///
/// 210 encodes the relationship between the encoder's internal MDCT scale and
/// the decoder's output scale, so it is part of the scale convention and has
/// to move with the headroom shift. Without this the encoder halves its
/// internal representation, lowers global_gain by 4 to match, and the decoder
/// faithfully reproduces a signal 6 dB quiet — which is exactly what the
/// failed headroom experiment measured: SNR 6.03 dB at shift 15, 2.50 at 14,
/// 1.16 at 13, each one precisely the SNR of a pure 2^(shift-16) gain error
/// and nothing else.
///
/// One quantizer step is 2^(1/4), so a factor of 2 in the domain is 4 steps.
inline int global_gain_offset() {
    return 4 * (REF_SHIFT - headroom_shift());
}

} // namespace rmps::scale
