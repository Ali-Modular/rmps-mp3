#pragma once

namespace rmps {
struct EncoderState;

namespace quant {

/// Legacy outer/inner loop orchestrator. Phase 2 baseline (no MS, no beam,
/// no HF stuff). Quantises every (ch, gr) into `s.l3_enc[ch][gr]`.
void iteration_loop(EncoderState& s);

} // namespace quant
} // namespace rmps
