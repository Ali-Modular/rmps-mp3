#pragma once
#include "control/transient_gate.hpp"
#include "core/bitstream.hpp"
#include "core/side_info.hpp"
#include "post/hf_adaptor.hpp"
#include "post/masking.hpp"
#include "rmps/config.hpp"
#include "rmps/stats.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace rmps {

/// Mirrors Encoder's EncoderMpeg.
struct MpegState {
  int bitr = 128; ///< kbps
  Emphasis emph = Emphasis::None;
  bool copyright = false;
  bool original = true;

  int version = 3; ///< MPEG_I=3 / II=2 / 2.5=0
  int layer = 1;   ///< Layer III bitstream value
  int granules_per_frame = 2;
  int padding = 0;
  int bits_per_frame = 0;
  int bits_per_slot = 8;
  double frac_slots_per_frame = 0.0;
  double slot_lag = 0.0;
  int whole_slots_per_frame = 0;
  int bitrate_index = 0;
  int samplerate_index = 0;
  bool crc = false;
  int ext = 0;
  int mode_ext = 0;
  Mode mode = Mode::JointStereo;
};

struct WaveState {
  int channels = 2;
  int samplerate = 44100;
};

/// Full encoder state. Single owner: Encoder::Impl.
struct EncoderState {
  WaveState wave;
  MpegState mpeg;

  Bitstream bs{4096};
  SideInfo side_info{};
  int sideinfo_len = 0;
  int mean_bits = 0;
  Scalefac scalefactor{};

  // ----- Cross-frame bit reservoir (main_data_begin) -----
  /// Continuous main_data byte stream produced by format_bitstream and
  /// drained into per-frame slots with a lag governed by the reservoir.
  std::vector<std::uint8_t> md_fifo;
  /// Bytes of md_fifo already written into emitted output-frame slots.
  std::size_t md_emitted = 0;
  /// Reservoir size in bytes = main_data_begin for the next frame. Bounded
  /// by the 9-bit (MPEG-1, ≤511) / 8-bit (MPEG-2, ≤255) field.
  int resv_bytes = 0;
  /// Frames whose header+side_info are built but whose main_data slot has not
  /// yet been filled from md_fifo (awaiting enough produced main_data).
  struct PendingFrame { std::vector<std::uint8_t> head; int slot; };
  std::vector<PendingFrame> pending_frames;

  // PCM input pointer (one per logical channel; same pointer with offset
  // 0/1 for interleaved inputs).
  std::array<const std::int16_t *, MAX_CHANNELS> buffer{nullptr, nullptr};
  std::array<std::size_t, MAX_CHANNELS> buffer_idx{0, 0};

  // Per-granule perceptual entropy (placeholder; reservoir uses it but the
  // simple encoder leaves it 0).
  std::array<std::array<double, MAX_GRANULES>, MAX_CHANNELS> pe{};

  // Quantized indices. [ch][gr][576]
  std::array<std::array<std::array<int, GRANULE_SAMPLES>, MAX_GRANULES>,
             MAX_CHANNELS>
      l3_enc{};

  // Subband sample cache. [ch][gr_plus_1][k][band]
  std::array<
      std::array<std::array<std::array<std::int32_t, SUBBANDS>, K_PER_GRANULE>,
                 3>,
      MAX_CHANNELS>
      l3_sb_sample{};

  // MDCT freq coefficients. [ch][gr][576]
  std::array<
      std::array<std::array<std::int32_t, GRANULE_SAMPLES>, MAX_GRANULES>,
      MAX_CHANNELS>
      mdct_freq{};

  int ResvSize = 0;
  int ResvMax = 0;

  // ----- loop tables (initialised once) -----
  std::array<double, 128> steptab{};
  std::array<std::int32_t, 128> steptabi{};
  std::array<int, 10000> int2idx{};

  // ----- subband state -----
  std::array<int, MAX_CHANNELS> subband_off{0, 0};
  std::array<std::array<std::int32_t, HAN_SIZE>, MAX_CHANNELS> subband_x{};
  std::array<std::array<std::int32_t, 64>, SUBBANDS> subband_fl{};

  // ----- MDCT tables -----
  std::array<std::array<std::int32_t, 36>, K_PER_GRANULE> mdct_cos_l{};

  // ----- per-iteration scratch (refreshed per granule) -----
  const std::int32_t *xr = nullptr;
  std::array<std::int32_t, GRANULE_SAMPLES> xrsq{};
  std::array<std::int32_t, GRANULE_SAMPLES> xrabs{};
  std::int32_t xrmax = 0;

  // ----- Post-processing modules -----
  std::unique_ptr<post::HFAdaptor> hf_adaptor;
  void hf_adaptor_apply();

  // ----- Pre-echo / transient gate (PCM-driven) -----
  control::TransientGate transient_gate;
  /// Per-granule effective HF cutoff (Hz). 22050 = no extra cutoff. Lower
  /// values are applied per-granule on top of the standard LPF when a
  /// transient is detected.
  std::array<double, MAX_GRANULES> transient_cutoff_hz {22050.0, 22050.0};
  std::array<bool,   MAX_GRANULES> transient_flags     {false, false};
  std::array<bool,   MAX_GRANULES> pre_window_flags    {false, false};
  std::array<bool,   MAX_GRANULES> attack_from_silence_flags {false, false};
  std::array<bool,   MAX_GRANULES> silent_flags              {false, false};

  // ----- 3D masking model -----
  /// Per-(ch, gr) masking thresholds, recomputed by iteration_loop after
  /// MDCT + post-processing and before beam_search.
  std::array<std::array<post::MaskingThresholds, MAX_GRANULES>, MAX_CHANNELS>
      mask_thresholds{};
  /// Per-channel temporal state for the masking model (post-mask carry).
  std::array<post::MaskingHistory, MAX_CHANNELS> mask_history{};
  /// Active masking model parameters (alpha, lambda_R, decay, ...).
  /// Refreshed each frame from the current bitrate / config.
  post::MaskingParams mask_params{};
  /// Config-provided alpha (0..1). Wins over bitrate-derived defaults.
  double cfg_mask_alpha = 0.3;
  /// Config-provided λ_R. Negative = auto-derive from bitrate.
  double cfg_mask_lambda_R = -1.0;
  /// Config-provided low-frequency strictness: lower the masking threshold
  /// of bands at/below cfg_lf_strict_cutoff_hz by this many dB (more bits to
  /// the bass). 0 = disabled.
  double cfg_lf_strict_db        = 10.0;
  double cfg_lf_strict_cutoff_hz = 1000.0;
  /// Linear pre-gain applied to PCM at subband load time. Compensates
  /// for the structural ~2 dB loudness shift in the analysis chain.
  double pre_gain = 1.0;
  /// Multiplier on the perceptual weight for HF SFBs (17–20). > 1
  /// forces the cost function to spend more bits on hi-hat sizzle.
  double hf_weight_boost = 1.0;

  /// Snapshot of the last successfully-encoded granule per channel,
  /// used by the outer_loop fallback path to inherit SFs / compress /
  /// scale / preflag instead of resetting to zeros. Without this,
  /// fallback granules emit default settings and the LF quant noise
  /// floor jumps relative to neighbouring beam_search granules,
  /// audible as random "ボフッ" sounds at high bitrate.
  struct LastGoodGranule {
      bool valid = false;
      int  compress = 0;
      int  scale    = 0;
      int  preflag  = 0;
      std::array<int, NUM_SFB_LONG + 1> sf {};
      std::array<int, 4> scfsi {};
  };
  std::array<LastGoodGranule, MAX_CHANNELS> last_good_granule {};

  // ----- Per-frame aggregate stats (populated by iteration_loop) -----
  std::vector<GranuleStats> frame_granule_stats;
  double frame_raw_signal_pow = 0.0;
  double frame_raw_dist = 0.0;
  double frame_w_signal_pow = 0.0;
  double frame_w_dist = 0.0;
  int frame_granule_count = 0;
  int frame_total_bits = 0;
  int frame_total_budget = 0;
  int frame_hf_rdo_saved = 0;
  int frame_transient_granules = 0;
  double frame_min_transient_cutoff_hz = 22050.0;
  /// Number of (gr, ch) pairs in this frame where beam_search returned
  /// no feasible path and the legacy outer_loop was used as fallback.
  /// Non-zero values indicate an LF quant-floor discontinuity (audible
  /// as random "boff" sounds), since outer_loop uses default SF=0 /
  /// compress=0 vs beam_search's optimised choices.
  int frame_outer_loop_fallbacks = 0;
};

} // namespace rmps
