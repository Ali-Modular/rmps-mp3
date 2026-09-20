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
  /// Window type this granule was coded with (BLOCK_LONG / START / SHORT /
  /// STOP). Carried so quality can be attributed per window — the transition
  /// windows are the ones whose real cost was never measured.
  int block_type = 0;
  int bits = 0; ///< part2 + part3 used
  int budget = 0;
  /// Fullness / Bleedless, dB, both <= 0 (0 = perfect). Split of the coding
  /// error into "did not keep" and "invented" — see beam_search.hpp.
  double fullness_db = 0.0;
  double bleedless_db = 0.0;
  double raw_sdr_db = 0.0;
  double perc_sdr_db = 0.0;
  int hf_rdo_bits_saved = 0;
};

/// What was actually written to the bitstream for one granule of one channel:
/// the side info and scalefactors a decoder reads. Filled for every granule
/// (long, short, beam or fallback), unlike `GranuleStats`, which describes the
/// search. Used by the GUI's analysis view.
struct GranuleSide {
  int gr = 0;
  int ch = 0;
  int block_type = 0;           ///< 0 long, 1 start, 2 short, 3 stop
  int global_gain = 0;          ///< as transmitted (step = 2^((gg-210)/4))
  int scalefac_scale = 0;
  int preflag = 0;
  int scalefac_compress = 0;
  int part2_bits = 0;           ///< scalefactor bits
  int part2_3_bits = 0;         ///< scalefactors + Huffman (incl. stuffing)
  int big_values = 0;           ///< pairs
  int count1 = 0;               ///< quadruples
  std::array<int, 3> table_select{0, 0, 0};
  int region0_count = 0;
  int region1_count = 0;
  int count1_table = 0;         ///< 0 = table A (32), 1 = table B (33)
  std::array<int, 3> subblock_gain{0, 0, 0};
  std::array<int, 4> scfsi{0, 0, 0, 0};
  std::array<std::int8_t, 21> sf_long{};                       ///< long blocks
  std::array<std::array<std::int8_t, 12>, 3> sf_short{};       ///< [window][sfb]
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
  /// Frame means of the per-granule Fullness / Bleedless, and the worse of
  /// the two — the quantity quality mode can target instead of SDR.
  double mean_fullness_db = 0.0;
  double mean_bleedless_db = 0.0;
  double fb_score_db = 0.0;
  int short_block_granules = 0;
  int transition_block_granules = 0;
  /// Number of granules this frame flagged as attack-from-silence (a
  /// subset of `transient_granules`). These get the deepest HF cut and
  /// the largest pre-mask penalty in the masking model.
  int attack_from_silence_granules = 0;
  /// Active beam-search cost mix and rate-distortion lambda.
  double mask_alpha = 0.0;
  double mask_lambda_R = 0.0;

  // --- Bitstream view of the frame (for analysis displays) ---
  int bitrate_kbps = 0;          ///< bitrate index chosen for this frame
  int samplerate = 0;            ///< coded rate
  bool ms_stereo = false;        ///< mode_extension M/S bit
  int reservoir_bits = 0;        ///< bit reservoir fill after this frame
  int reservoir_max_bits = 0;
  int overflow_bits = 0;         ///< discarded as stuffing (bank full)
  double lowpass_mid_hz = 0.0;   ///< Mid lowpass in force (<= 0 if none)
  std::vector<GranuleSide> side; ///< [gr * channels + ch]
};

} // namespace rmps
