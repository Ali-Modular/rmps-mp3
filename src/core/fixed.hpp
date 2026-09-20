#pragma once
#include <cstdint>

namespace rmps::fx {

/// Wrap to signed 32-bit (mirrors Encoder's `int32`).
///
/// **Do not use this to store a value that can exceed int32.** It wraps, so an
/// overflow becomes a sign inversion rather than a clip — which is how this
/// encoder produced its "boff" thumps: the M/S rotation overflowed at the kick
/// fundamental and turned +2.36e9 into −1.94e9. Every site that stores a
/// computed value now uses `sat32`; `i32` remains only inside `mul`/`mulr`,
/// where the >>32 shift bounds the result at 2^30 and no overflow is possible.
inline std::int32_t i32(std::int64_t x) noexcept {
  return static_cast<std::int32_t>(
      static_cast<std::uint32_t>(x & 0xFFFFFFFFLL));
}

/// Saturating store to signed 32-bit — the safe default for any accumulator,
/// rotation or gain whose result may not fit. Clipping costs a bounded amount
/// of one coefficient's amplitude; wrapping costs the whole waveform.
///
/// This exists because wrapping cost a real defect. `ms_stereo` stored
/// M = (L+R)/√2 through `i32`, and on loud centred bass — where L ≈ R and the
/// rotation multiplies by √2 — it overflowed: 549 coefficients on one 235 s
/// program, **every one at bin 1**, the kick fundamental, turning
/// +2,358,746,202 into −1,936,221,094. A full-scale sign inversion in a kick's
/// bass is the thump users report as "boff", and nothing downstream can see it
/// because it corrupts the quantizer's input. `xrsq` had the same defect one
/// count past the top (`v*v >> 31` reaches 2^31 at full scale), handing the
/// psychoacoustic model a negative energy for the loudest bin.
///
/// Audited across the encoder afterwards on two real programs: those two were
/// the only sites that actually fired, but every store now saturates so the
/// class cannot come back.
inline std::int32_t sat32(std::int64_t x) noexcept {
  if (x > 2147483647LL) return 2147483647;
  if (x < -2147483648LL) return -2147483647 - 1;
  return static_cast<std::int32_t>(x);
}

/// Same, from a double. Clamping *before* the cast also removes the undefined
/// behaviour of converting an out-of-range floating value to an integer.
inline std::int32_t sat32(double x) noexcept {
  if (!(x > -2147483649.0)) return -2147483647 - 1; // also catches NaN
  if (x >= 2147483647.0) return 2147483647;
  if (x <= -2147483648.0) return -2147483647 - 1;
  return static_cast<std::int32_t>(x);
}

/// Q31 multiply, truncating: (a * b) >> 32.
inline std::int32_t mul(std::int32_t a, std::int32_t b) noexcept {
  return i32(static_cast<std::int64_t>(a) * static_cast<std::int64_t>(b) >> 32);
}

/// Q31 multiply with rounding: ((a * b) + 2^31) >> 32.
inline std::int32_t mulr(std::int32_t a, std::int32_t b) noexcept {
  const std::int64_t p =
      static_cast<std::int64_t>(a) * static_cast<std::int64_t>(b) +
      static_cast<std::int64_t>(0x80000000LL);
  return i32(p >> 32);
}

/// Q31 square + bias used by Encoder's mulsr macro.
inline std::int32_t mulsr(std::int32_t a, std::int32_t b) noexcept {
  const std::int64_t p =
      static_cast<std::int64_t>(a) * static_cast<std::int64_t>(b) +
      static_cast<std::int64_t>(0x40000000LL);
  return i32(p >> 31);
}

} // namespace rmps::fx
