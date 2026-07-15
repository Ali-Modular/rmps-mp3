#pragma once

namespace rmps {
struct EncoderState;

namespace quant {

/// Distortion / signal stats of a quantized short granule, on the same scale
/// as BeamResult so the A/B block-switching compare in run_frame can weigh a
/// short-block frame against the forced-long alternative. cum_dist is the
/// perceptually (A-)weighted quantization noise plus the λ_R rate term;
/// without it the short granule contributes zero cost and the switching path
/// gets a free pass on the bass it coarsens.
struct ShortBlockResult {
  bool ok = false;
  double raw_signal_pow = 0.0;
  double raw_dist = 0.0;
  double weighted_signal_pow = 0.0;
  double cum_dist = 0.0;
};

/// Quantize one short-block (block_type == BLOCK_SHORT) granule.
///
/// Mirrors the role of beam_search() / outer_loop() for short blocks: it writes
/// `ix[0..575]` (in short transmission order — already reordered by the MDCT),
/// the GrInfo side-info fields (global_gain via quantizerStepSize,
/// scalefac_compress, subblock_gain, big_values/count1/region/table_select,
/// part2_length, part2_3_length) and the per-window short scalefactors
/// s.scalefactor.s[gr][ch][w][sfb]. Returns true on success.
///
/// The short-block bitstream rules differ from long blocks: scfsi and preflag
/// do not apply, the big-values region split is fixed at coefficient 36, and
/// scalefactors are per (window, sfb).
ShortBlockResult short_block_quantize(int max_bits, int* ix, int gr, int ch,
                                      EncoderState& s);

} // namespace quant
} // namespace rmps
