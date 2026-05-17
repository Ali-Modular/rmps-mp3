#pragma once

namespace rmps {
struct EncoderState;

namespace l3 {

/// Restore signs, write header + side info + main data into `s.bs`.
void format_bitstream(EncoderState& s);

} // namespace l3
} // namespace rmps
