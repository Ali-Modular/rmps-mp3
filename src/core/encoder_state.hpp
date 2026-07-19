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
  std::array<double, 256> steptab{};
  std::array<std::int32_t, 256> steptabi{};
  std::array<int, 262144> int2idx{};

  // ----- scratch buffers for best_subdivide_rd -----
  struct SubdivideScratch {
    std::array<std::array<std::uint32_t, 289>, 13> pb_s;
    std::array<std::uint32_t, 289> pb16;
    std::array<std::uint32_t, 289> pb24;
    std::array<std::uint16_t, 289> pesc;
    std::array<std::uint16_t, 289> pmax;
  } subdivide_scratch{};


  // ----- subband state -----
  std::array<int, MAX_CHANNELS> subband_off{0, 0};
  std::array<std::array<std::int32_t, HAN_SIZE>, MAX_CHANNELS> subband_x{};
  std::array<std::array<std::int32_t, 64>, SUBBANDS> subband_fl{};

  // ----- MDCT tables -----
  std::array<std::array<std::int32_t, 36>, K_PER_GRANULE> mdct_cos_l{};
  // Window-folded 36-point bases for the start (long->short) and stop
  // (short->long) transition blocks. Same layout/semantics as mdct_cos_l
  // (Q31, [output 0..17][input 0..35]); only the window shape differs.
  std::array<std::array<std::int32_t, 36>, K_PER_GRANULE> mdct_cos_start{};
  std::array<std::array<std::int32_t, 36>, K_PER_GRANULE> mdct_cos_stop{};
  // Window-folded 12-point short MDCT basis: [output 0..5][input 0..11], Q31.
  // Applied independently to each of the 3 short sub-windows of a subband.
  std::array<std::array<std::int32_t, 12>, 6> mdct_cos_s{};
  // Short-block reorder: transmission index (0..575) -> natural position
  // (sb*18 + w*6 + k) produced by the per-subband short MDCT. Built at init.
  std::array<int, GRANULE_SAMPLES> short_reorder{};

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
  /// Per-granule: attack carries significant HF content (worth a short block).
  /// Low-frequency transients (kicks) stay long to preserve bass resolution.
  std::array<bool,   MAX_GRANULES> hf_transient_flags        {false, false};

  // ----- Per-(ch, gr) block type (window switching) -----
  /// Decided per frame by control::decide_block_types before MDCT, consumed by
  /// mdct::analyse (window choice) and iteration_loop (quantizer choice).
  std::array<std::array<int, MAX_GRANULES>, MAX_CHANNELS> block_type {};
  /// Last granule's block type per channel, carried across frames so the
  /// start/stop transition state machine stays continuous.
  std::array<int, MAX_CHANNELS> prev_block_type {BLOCK_LONG, BLOCK_LONG};
  /// Master enable for window switching (Config::block_switching). When false
  /// every granule stays BLOCK_LONG and the encoder is bit-identical to before.
  bool cfg_block_switching = true;

  // ----- Sub-window energy profile (from TransientGate) -----
  /// Per-granule sub-window energies (3 × 192-sample windows per granule).
  /// Used by the masking model to estimate pre-echo exposure.
  std::array<std::array<double, 3>, MAX_GRANULES> sub_window_energies{};
  /// Sub-window index (0–2) where the attack fires, or -1 for no attack.
  std::array<int,    MAX_GRANULES> attack_sub_index{-1, -1};
  /// Energy contrast ratio (attack / preceding sub-window).
  std::array<double, MAX_GRANULES> energy_contrast{1.0, 1.0};

  /// When the current frame's gr=0 is a transient, the previous frame's
  /// last granule should be treated as a pre_window (but was already
  /// encoded). This flag tells the masking model to inject an artificial
  /// low threshold into the history so the post-mask carry bleeds into
  /// the current frame's thresholds — an approximate retroactive fix.
  /// Last granule of the PREVIOUS frame carried an attack (captured in
  /// encoder.cpp BEFORE analyse_frame overwrites the flag arrays —
  /// transient_gate.prev_frame_last_decision() is already rewritten to the
  /// current frame by encode time, so it cannot serve this purpose).
  /// A sub-window-2 attack lands, after the polyphase delay, inside the
  /// NEXT frame's gr0 MDCT window; beam_search's temporal shaping and the
  /// iteration_loop attack surge both need to know.
  bool prev_frame_attack     = false;
  int  prev_frame_attack_sub = -1;
  /// Per-granule multiplier on mask_params.lambda_R, set by iteration_loop
  /// before each quantizer call. Couples the bit-reservoir state into the
  /// beam search's rate-distortion cost: <1 on attack granules (spend the
  /// bank), >1 when the bank is lean (refill it).
  double lambda_granule_scale = 1.0;
  bool cross_frame_pre_window = false;
  /// Energy contrast of the cross-frame boundary (current gr=0 attack
  /// vs previous frame's last sub-window).
  double cross_frame_energy_contrast = 1.0;

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
  /// Boost cap in dB for the transparent zero fill ("black-spot boost",
  /// CLI --hf-zero-boost). Compressive law: zeroed coefficients near the
  /// dead-zone boundary get ≈0 dB lift, quieter ones half their depth in
  /// dB, saturating at this cap; realised as |ix|=1 promotions whose
  /// density follows the curve, energy-capped at the band's actual
  /// deficit so it cannot read as an HF boost. 0 = disabled.
  double cfg_hf_zero_boost_db = 0.0;
  /// 1st-order error-feedback noise-shaping coefficient used by the
  /// per-band shaping quantizer (0..0.95). 0 = no shaping. Sourced from
  /// Config::noise_shaping_alpha (CLI --shaping); default 0.5.
  double cfg_noise_shaping_alpha = 0.5;
  /// Waveform preservation regularization strength (0..1.0).
  double cfg_waveform_reg = 0.05;

  /// Adaptive joint-stereo: when true, the encoder decides per frame whether
  /// M/S coding actually compresses better than plain L/R and switches
  /// mode_ext accordingly (2 = MS on, 0 = MS off). Always-on MS wastes bits
  /// on weakly/anti-correlated material where E[S] ≈ E[M].
  bool cfg_adaptive_ms = true;
  /// Previous frame's MS decision (hysteresis anchor to stop flip-flopping
  /// at the decision boundary, which would otherwise modulate the stereo
  /// image frame-to-frame).
  bool ms_prev_frame = true;

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
