#pragma once

namespace rmps {
struct EncoderState;

namespace mdct {

/// Initialise MDCT cosine table (called once).
void initialise(EncoderState& s);

/// Full per-frame analysis: PCM (via subband) -> 32 subbands -> 18-point MDCT
/// per subband -> 576 freq coefficients, with anti-aliasing butterflies.
/// `stride` is 1 for planar buffers, 2 for LR-interleaved.
void analyse(EncoderState& s, int stride);

} // namespace mdct
} // namespace rmps
