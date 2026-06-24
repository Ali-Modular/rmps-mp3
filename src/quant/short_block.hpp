#pragma once

namespace rmps {
struct EncoderState;

namespace quant {

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
bool short_block_quantize(int max_bits, int* ix, int gr, int ch, EncoderState& s);

} // namespace quant
} // namespace rmps
