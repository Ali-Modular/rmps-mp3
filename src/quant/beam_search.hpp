#pragma once

namespace rmps {
struct EncoderState;

namespace quant {

struct BeamResult {
    bool   ok = false;
    int    part2_3_length = 0;
    double cum_dist       = 0.0;
    double weighted_signal_pow = 0.0;
    double raw_signal_pow      = 0.0;
    double raw_dist            = 0.0;
    int    hf_rdo_bits_saved   = 0;
};

/// Run the beam-search trellis quantization for one (gr, ch). Returns
/// BeamResult::ok=false if no feasible path was found within the bit budget
/// (caller may fall back to legacy outer/inner loop). Writes results into
/// s.side_info / s.scalefactor / ix as a side-effect.
BeamResult beam_search(int max_bits, int* ix, int gr, int ch, EncoderState& s,
                       const int* gr0_sf = nullptr);

} // namespace quant
} // namespace rmps
