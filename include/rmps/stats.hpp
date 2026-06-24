#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace rmps {

/// Per-granule encode stats.
struct GranuleStats {
  int gr = 0;
  int ch = 0;
  int gain = 0;
  int preflag = 0;
  int scale = 0;
  int compress = 0;
  std::array<int, 4> scfsi_pat{0, 0, 0, 0};
  int bits = 0; ///< part2 + part3 used
  int budget = 0;
  double raw_sdr_db = 0.0;
  double perc_sdr_db = 0.0;
  int hf_rdo_bits_saved = 0;
};

/// HFAdaptor visible state.
struct HFAdaptorState {
  double atten_db = 0.0;
  bool transient = false;
  double shelf_hz = 8000.0;
};

/// PID/ABR controller state (only meaningful when ABR is active).
struct AbrState {
  double target_kbps = 0.0;
  double avg_kbps = 0.0;
  double decision_kbps = 0.0;
  double err = 0.0;
  double u = 0.0;
  double integral = 0.0;
  double min_kbps = 0.0;
  double max_kbps = 0.0;
  long frames = 0;
};

/// Aggregated per-frame stats; emitted after every frame.
struct FrameStats {
  std::vector<GranuleStats> granules;
  int total_bits = 0;
  int total_budget = 0;
  double mean_raw_sdr_db = 0.0;
  double mean_perc_sdr_db = 0.0;
  int hf_rdo_bits_saved = 0;
  HFAdaptorState hf{};
  double lpf_side_hz = 0.0;
  /// PCM-driven transient gate: how many granules in this frame were
  /// flagged as containing a transient, and the most aggressive cutoff
  /// applied to any of them. 22050 = no cutoff applied.
  int transient_granules = 0;
  double transient_cutoff_hz = 20000.0;
  /// Window-switching usage this frame: number of (ch, gr) granules that used
  /// a short block (block_type 2) and the number of transition blocks
  /// (start/stop). 0/0 means the whole frame stayed long.
  int short_block_granules = 0;
  int transition_block_granules = 0;
  /// Number of granules this frame flagged as attack-from-silence (a
  /// subset of `transient_granules`). These get the deepest HF cut and
  /// the largest pre-mask penalty in the masking model.
  int attack_from_silence_granules = 0;
  /// Active beam-search cost mix and rate-distortion lambda.
  double mask_alpha = 0.0;
  double mask_lambda_R = 0.0;
};

} // namespace rmps
