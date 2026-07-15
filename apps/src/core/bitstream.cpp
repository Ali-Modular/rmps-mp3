#include "core/bitstream.hpp"

#include <cassert>

namespace rmps {

Bitstream::Bitstream(std::size_t initial_size)
    : buffer_(initial_size) {}

void Bitstream::reset() {
    position_   = 0;
    cache_      = 0;
    cache_room_ = 32;
}

void Bitstream::grow() {
    const std::size_t new_size = buffer_.size() + buffer_.size() / 2 + 16;
    buffer_.resize(new_size);
}

void Bitstream::putbits(std::uint32_t val, int N) {
    if (N == 0) return;
    assert(N > 0 && N <= 32);

    // Keep only the lower N bits.
    val &= (N == 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);

    if (cache_room_ > N) {
        cache_room_ -= N;
        cache_ |= (val << cache_room_);
        return;
    }

    // Fill what's left of the cache, flush, then continue with the remainder.
    int remaining = N - cache_room_;
    cache_ |= (val >> remaining);

    if (position_ + 4 >= buffer_.size()) grow();
    buffer_[position_    ] = static_cast<std::uint8_t>((cache_ >> 24) & 0xFF);
    buffer_[position_ + 1] = static_cast<std::uint8_t>((cache_ >> 16) & 0xFF);
    buffer_[position_ + 2] = static_cast<std::uint8_t>((cache_ >>  8) & 0xFF);
    buffer_[position_ + 3] = static_cast<std::uint8_t>( cache_        & 0xFF);
    position_ += 4;

    cache_room_ = 32 - remaining;
    cache_      = remaining ? (val << cache_room_) : 0;
}

void Bitstream::flush_bytes() {
    int used = 32 - cache_room_;          // bits currently held in cache (MSB-aligned)
    while (used >= 8) {
        if (position_ >= buffer_.size()) grow();
        buffer_[position_++] = static_cast<std::uint8_t>((cache_ >> 24) & 0xFF);
        cache_    <<= 8;
        used       -= 8;
        cache_room_ += 8;
    }
}

std::size_t Bitstream::bit_count() const noexcept {
    return position_ * 8 + static_cast<std::size_t>(32 - cache_room_);
}

} // namespace rmps
