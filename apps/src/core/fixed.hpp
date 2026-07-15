#pragma once
#include <cstdint>

namespace rmps::fx {

/// Wrap to signed 32-bit (mirrors Encoder's `int32`).
inline std::int32_t i32(std::int64_t x) noexcept {
  return static_cast<std::int32_t>(
      static_cast<std::uint32_t>(x & 0xFFFFFFFFLL));
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
