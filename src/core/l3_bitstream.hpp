#pragma once

namespace rmps {
struct EncoderState;

namespace l3 {

/// Restore signs, encode this frame's main_data into the cross-frame
/// reservoir FIFO, and emit any now-complete output frames into `s.bs`.
void format_bitstream(EncoderState& s);

/// End of stream: emit all buffered frames, zero-padding under-filled slots.
/// Output goes to `s.bs` (read it the same way as after format_bitstream).
void flush_bitstream(EncoderState& s);

} // namespace l3
} // namespace rmps
