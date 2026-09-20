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
    /// A-weighted distortion power, the companion of `weighted_signal_pow`.
    /// `cum_dist` is a *cost* in loudness units and cannot be divided into a
    /// signal power to make an SDR; this can.
    double weighted_dist       = 0.0;
    /// Fullness / Bleedless for this granule, dB, both <= 0 with 0 perfect.
    /// Fullness measures source energy that did not survive; Bleedless,
    /// energy that was never there. Kept apart on purpose — SDR sums them and
    /// then cannot tell a hole from a fabrication.
    double fullness_db         = 0.0;
    double bleedless_db        = 0.0;
    /// The A-weighted powers `fullness_db` / `bleedless_db` are built from:
    /// source energy the coding failed to keep, and energy it invented.
    /// Exported so a *frame* score can be one ratio of summed powers instead
    /// of a mean of per-granule decibels — see the aggregation in encoder.cpp.
    double weighted_def        = 0.0;
    double weighted_exc        = 0.0;
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
