#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rmps {

/// MSB-first bit packer that owns a growable byte buffer. Mirrors the
/// behaviour of Encoder's `Bitstream`: a 32-bit cache holds pending high bits
/// and is flushed in big-endian order whenever it fills.
class Bitstream {
public:
  explicit Bitstream(std::size_t initial_size = 4096);

  /// Hard reset: clears buffer position AND cache. Use only at stream start.
  void reset();

  /// Consume the currently flushed bytes (position_ <- 0) but keep the
  /// internal cache. Lets a multi-frame stream stay bit-continuous across
  /// frame boundaries even when bits_per_frame is not a multiple of 32.
  void consume_position() noexcept { position_ = 0; }

  /// Write the lower N bits of `val`, MSB first. N must be 0..32.
  void putbits(std::uint32_t val, int N);

  /// Number of bits written since the last reset().
  std::size_t bit_count() const noexcept;

  /// Current byte position (cache not yet flushed).
  std::size_t byte_position() const noexcept { return position_; }

  /// Borrow the underlying bytes [0, byte_position()). Valid until the next
  /// putbits / reset.
  const std::uint8_t *data() const noexcept { return buffer_.data(); }

private:
  void grow();

  std::vector<std::uint8_t> buffer_;
  std::size_t position_ = 0; ///< bytes already flushed
  std::uint32_t cache_ = 0;  ///< pending high bits (MSB-first)
  int cache_room_ = 32;      ///< unused bits left in cache_
};

} // namespace rmps
