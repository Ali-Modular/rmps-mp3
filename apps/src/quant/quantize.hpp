#pragma once

namespace rmps {
struct EncoderState;
struct GrInfo;

namespace quant {

/// Build loop tables (steptab / steptabi / int2idx). Call once.
void initialise(EncoderState& s);

/// Master quantizer: write `ix[0..575]` and return max-abs index.
int quantize(int* ix, int stepsize, EncoderState& s, int gr, int ch);

void calc_runlen      (int* ix, GrInfo& gi);
int  count1_bitcount  (int* ix, GrInfo& gi);
void subdivide        (GrInfo& gi, EncoderState& s);
/// RD-optimal big-values region split: exhaustively searches
/// (region0_count, region1_count) over scalefactor-band boundaries and picks
/// the joint region/table assignment minimising the big-values Huffman bits.
/// Requires calc_runlen() to have set gi.big_values. Writes region counts,
/// addresses and table_select into `gi` and returns the big-values bit count.
/// Strictly never produces more bits than the fixed-table subdivide() path.
int  best_subdivide_rd(int* ix, GrInfo& gi, EncoderState& s);
void bigv_tab_select  (int* ix, GrInfo& gi);
int  bigv_bitcount    (int* ix, const GrInfo& gi);
int  count_bit        (int* ix, int start, int end, int table);
int  ix_max           (int* ix, int begin, int end);
int  new_choose_table (int* ix, int begin, int end);

int  bin_search_StepSize(int desired_rate, int* ix, GrInfo& gi, EncoderState& s, int gr, int ch);
int  inner_loop         (int* ix, int max_bits, GrInfo& gi, EncoderState& s, int gr, int ch);
int  part2_length       (int gr, int ch, EncoderState& s);
int  outer_loop         (int max_bits, int* ix, int gr, int ch, EncoderState& s);

} // namespace quant
} // namespace rmps
