#pragma once

namespace rmps {
struct EncoderState;
struct GrInfo;

namespace reservoir {

int  max_reservoir_bits(double pe_val, EncoderState& s);
void adjust(const GrInfo& gi, EncoderState& s);
void frame_end(EncoderState& s);

} // namespace reservoir
} // namespace rmps
