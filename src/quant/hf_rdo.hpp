#pragma once
#include "core/side_info.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace rmps {
struct EncoderState;

namespace quant {

/// Per-coefficient HF rate-distortion polish. Mutates ix / bits / dist_band
/// in place; returns total bits saved (always >= 0).
int hf_rdo(int* ix_full, int& cur_bits,
           std::vector<double>& dist_band,
           int huff_budget,
           int gain, int scale, int preflag,
           const std::array<const std::int32_t*, NUM_SFB_LONG>& xr_bands,
           const std::array<const std::int32_t*, NUM_SFB_LONG>& xrabs_bands,
           const std::array<int, NUM_SFB_LONG>& band_lens,
           const std::array<int, 23>& band_table,
           const std::array<double, 21>& weights,
           EncoderState& s,
           GrInfo& tmp_gi,
           int hf_start_sfb);

}}
