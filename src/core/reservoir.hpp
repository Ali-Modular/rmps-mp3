#pragma once

namespace rmps {
struct EncoderState;
struct GrInfo;

namespace reservoir {

int  max_reservoir_bits(double pe_val, EncoderState& s);
/// Bitrate-tapered reservoir cap (bits): full ResvMax at >=96 kbps, ramps
/// to 0 by 72 kbps to stop low-rate cross-frame bit swings pumping the bass.
int  effective_resv_max(const EncoderState& s);
void adjust(const GrInfo& gi, EncoderState& s);
void frame_end(EncoderState& s);

} // namespace reservoir
} // namespace rmps
