#pragma once

namespace rmps {
struct EncoderState;

namespace subband {

/// One-shot table build (call from Encoder constructor).
void initialise(EncoderState& s);

/// Polyphase analysis: read 32 PCM samples from buffer[ch] (stride is the
/// interleave step, 1 = planar, 2 = LR-interleaved) and produce 32 subband
/// samples in `out[0..31]`.
void window_filter(int ch, int* out, EncoderState& s, int stride);

/// Convenience overload with std::int32_t output.
void window_filter(int ch, int* out, EncoderState& s);

/// Print fixed-point headroom statistics for the analysis chain
/// (RMPS_FX_PROBE): how often each conversion saturates and how far past the
/// limit it reached.
void fx_probe_report();

} // namespace subband

} // namespace rmps
