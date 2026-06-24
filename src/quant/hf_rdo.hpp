#pragma once
#include "core/side_info.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace rmps {
struct EncoderState;

namespace quant {

/// Per-coefficient HF rate-distortion polish. Mutates ix / bits /
/// dist_band / raw_band in place; returns total bits saved (always >= 0).
/// `raw_band` is kept in step with the unweighted noise power per band so
/// callers can still report raw SDR after this pass.
/// `hf_end_sfb` is the exclusive upper bound. Callers protecting the
/// topmost SFB (noise-filling mode) pass `NUM_SFB_LONG - 1`.
int hf_rdo(int* ix_full, int& cur_bits,
           std::vector<double>& dist_band,
           std::vector<double>& raw_band,
           int huff_budget,
           int gain, int scale, int preflag,
           const std::array<const std::int32_t*, NUM_SFB_LONG>& xr_bands,
           const std::array<const std::int32_t*, NUM_SFB_LONG>& xrabs_bands,
           const std::array<int, NUM_SFB_LONG>& band_lens,
           const std::array<int, 23>& band_table,
           const std::array<double, 21>& weights,
           const std::array<double, NUM_SFB_LONG>& tonality,
           EncoderState& s,
           GrInfo& tmp_gi,
           int hf_start_sfb,
           int hf_end_sfb = NUM_SFB_LONG);

}}
