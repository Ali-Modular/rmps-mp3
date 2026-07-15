#pragma once
#include <array>
#include <cstdint>

namespace rmps {
inline constexpr int NUM_SFB_LONG_ = 21;

namespace post {

/// Returns A-weighted per-SFB perceptual weights.
/// `is_side_channel == true` returns the MS-relaxed weights for the S
/// channel (image-band 500..3000 Hz x0.60, outer x0.30).
const std::array<double, NUM_SFB_LONG_>&
perceptual_weights(int samplerate_index, int samplerate, bool is_side_channel);

} // namespace post
} // namespace rmps
