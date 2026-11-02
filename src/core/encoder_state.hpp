#pragma once
#include "bend/bend.hpp"
#include "control/transient_gate.hpp"
#include "core/bitstream.hpp"
#include "core/side_info.hpp"
#include "post/hf_adaptor.hpp"
#include "post/masking.hpp"
#include "rmps/config.hpp"
#include "rmps/stats.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rmps {

/// The quantizer's magnitude map: i^(3/4), rounded with the encoder's 0.0946
/// bias, for i in [0, 262144). Hosted builds keep it as a 1 MB table.
///
/// RMPS_COMPACT_TABLES is for targets with far less RAM than that (the Daisy
/// has 1 MB of internal SRAM in all). It keeps the table only below
/// kSmall, 8 kB, which is where nearly every lookup lands once the search
/// has found its step, and computes the rest from two single-precision
/// square roots. The table part is exact. Measured on x86, the computed part
/// is off by one at a rounding boundary on 87 of its inputs, and all 87 are
/// above kSmall, so computing all of it would give the same output.
/// Computing every entry, without the table, cost the Cortex-M7's short-block
/// search half its time: the search quantizes all 576 lines on every step.
struct Int2Idx {
#if defined(RMPS_COMPACT_TABLES)
  static constexpr std::size_t kSmall = 4096;
  void build() noexcept {
    for (std::size_t i = 0; i < kSmall; ++i)
      small[i] = static_cast<std::uint16_t>(std::pow(static_cast<double>(i), 0.75) - 0.0946 + 0.5);
  }
  int operator[](std::size_t i) const noexcept {
    if (i < kSmall)
      return small[i];
    const float r = std::sqrt(static_cast<float>(i));
    return static_cast<int>(r * std::sqrt(r) - 0.0946f + 0.5f);
  }

private:
  std::array<std::uint16_t, kSmall> small{};
#else
  void build() noexcept {
    for (std::size_t i = 0; i < t.size(); ++i)
      t[i] = static_cast<int>(std::pow(static_cast<double>(i), 0.75) - 0.0946 + 0.5);
  }
  int operator[](std::size_t i) const noexcept { return t[i]; }

private:
  std::array<int, 262144> t{};
#endif
};

struct MpegState;

/// Sets the slots (bytes) per frame for the current bitrate, as a whole part
/// and a fraction, from `samplerate`. In integers: in floating point
/// 1152/32000 is not exact, and at 32 kHz 56, 96, 192 and 224 kbps came out a
/// hair under a whole number of slots. Those frames were then signalled as
/// padded while carrying the unpadded length, and every decoder lost sync.
inline void set_frame_slots(MpegState &m, int samplerate);

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

inline void set_frame_slots(MpegState &m, int samplerate) {
  const long long num = static_cast<long long>(m.granules_per_frame) * 576 * 1000 * m.bitr;
  const long long den = static_cast<long long>(samplerate) * m.bits_per_slot;
  m.whole_slots_per_frame = static_cast<int>(num / den);
  m.frac_slots_per_frame = static_cast<double>(num % den) / static_cast<double>(den);
}

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

  // The previous granule's quantized indices, per channel, and whether they
  // are usable. Only meaningful between two consecutive LONG granules: the
  // short-block layout interleaves three sub-windows, so line i does not name
  // the same frequency in both. Read by the quantizer's zero hysteresis.
  std::array<std::array<int, GRANULE_SAMPLES>, MAX_CHANNELS> prev_l3_enc{};
  std::array<bool, MAX_CHANNELS> prev_l3_enc_long{};
  /// The stereo mode those indices were coded in. Channel 1 is Side under
  /// M/S and Right under L/R, and `apply_ms_stereo` re-decides per frame, so
  /// across a mode flip the two granules are not the same signal at all.
  int prev_l3_enc_mode_ext = -1;
  /// Source band energies of the previous granule, per channel — what the
  /// band hole guard compares against to decide whether the source was steady.
  /// Valid under the same conditions as `prev_l3_enc` (see prev_l3_enc_long).
  std::array<std::array<double, NUM_SFB_LONG>, MAX_CHANNELS> prev_e_band{};
  /// Reconstructed (decoded) band energies of the previous granule, per
  /// channel, same units as `prev_e_band`. Zeroed at the start of every beam
  /// search and written only when it succeeds, so a fallback granule leaves no
  /// stale reference behind for the next granule's hole rescue.
  std::array<std::array<double, NUM_SFB_LONG>, MAX_CHANNELS> prev_erec_band{};
  /// Mid lowpass hold: the cutoff actually applied, and the run of frames the
  /// scan has asked to move it in one direction (see hold_mid_cutoff).
  double lpf_mid_applied_hz = -1.0;
  int lpf_mid_pending_dir = 0;
  int lpf_mid_pending_n = 0;
  double lpf_mid_pending_hz = 0.0;
  int lpf_mid_cut_age = 1 << 20;   ///< frames since the last applied cut
  /// Hit / steady HF matching (see iteration_loop.cpp). A slow, power-
  /// weighted average per channel and band of what the encoder reconstructs
  /// (er) against what the source had (es), taken on steady long granules.
  /// Their ratio is the coding gain steady material actually gets; zero until
  /// measured, which means "no attenuation".
  std::array<std::array<double, NUM_SFB_LONG>, MAX_CHANNELS> hf_steady_er{};
  std::array<std::array<double, NUM_SFB_LONG>, MAX_CHANNELS> hf_steady_es{};
  /// Granules since the last detected attack, and which granules of the frame
  /// being coded fall inside a hit (the attack and the granules after it).
  int granules_since_hit = 1 << 20;
  std::array<bool, MAX_GRANULES> hit_window{};
  /// Debug snapshot taken where beam_search fixes part2_3_length, compared
  /// by the main-data writer: if the data or the run lengths change between
  /// counting and writing, the side info is stale.
  std::array<std::array<std::array<int, GRANULE_SAMPLES>, MAX_CHANNELS>,
             MAX_GRANULES> dbg_fin_ix{};
  std::array<std::array<int, MAX_CHANNELS>, MAX_GRANULES> dbg_fin_bigv{},
      dbg_fin_count1{}, dbg_fin_p23{};
  std::array<std::array<bool, MAX_CHANNELS>, MAX_GRANULES> dbg_fin_valid{};

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

  // Synthesized HF MDCT freq coefficients for bandwidth extension. [ch][gr][576]
  std::array<
      std::array<std::array<std::int32_t, GRANULE_SAMPLES>, MAX_GRANULES>,
      MAX_CHANNELS>
      mdct_freq_ext{};

  int ResvSize = 0;
  int ResvMax = 0;
  /// Bits `reservoir::frame_end` discarded this frame because the bank was
  /// already full — coded as stuffing, i.e. paid for in the stream and carrying
  /// nothing. Excludes the 0-7 bits of byte alignment, which are unavoidable.
  int frame_overflow_bits = 0;

  // ----- loop tables (initialised once) -----
  std::array<double, 256> steptab{};
  std::array<std::int32_t, 256> steptabi{};
  Int2Idx int2idx{};

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
  /// Last sub-window energy of the previous frame's last granule, kept before
  /// `sub_window_energies` is overwritten, so granule 0 can measure its rise
  /// against what actually preceded it.
  double prev_frame_last_sub_energy = 0.0;
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

  /// The encode's operating point in kbps, as opposed to this frame's rate.
  ///
  /// Every rate-keyed *policy* in the encoder — the Side lowpass, the M/S band
  /// pruning, the masking alpha and lf_strict ramps, the reservoir taper, the
  /// HF boost — was written against `mpeg.bitr`, which under CBR is the same
  /// thing. Under quality-targeted VBR it is not: the rate swings 129-320 kbps
  /// within a single 256 kbps encode (measured, 5 % of frames under 128), and
  /// an easy frame that needed few bits was then coded with the *bandwidth and
  /// stereo rules of a 128 kbps file*. That is what made quiet passages weak,
  /// and it is why none of the rate-controller experiments moved them: the
  /// controller was never the problem, the policies keyed to its output were.
  ///
  /// Set to max(this frame's rate, the controller's running average), so a
  /// frame that got more bits than the encode's average still uses its own
  /// richer policy, and one that got fewer is not demoted below the encode it
  /// belongs to. Under CBR the two are equal and nothing changes.
  double policy_kbps = 0.0;

  /// `policy_kbps` as an int, falling back to the frame rate when no rate
  /// controller is running. Use this for policy decisions; use `mpeg.bitr`
  /// only where the actual bit budget of this frame is what matters.
  int policy_bitr() const {
    return (policy_kbps > 0.0) ? static_cast<int>(policy_kbps + 0.5)
                               : mpeg.bitr;
  }

  // ----- One-frame lookahead (filled by Encoder::encode) -----
  /// True when the frame after this one has already been seen. False on the
  /// final frame, where there is nothing ahead to plan for.
  /// Frame counter, for frame-scoped debug dumps only.
  long long dbg_frame_no = 0;

  bool lookahead_valid = false;
  /// The next frame contains a sharp energy step — it will want bits.
  bool lookahead_attack = false;
  /// The next frame's attack lands in *its granule 0*. The START window that
  /// must precede a SHORT belongs to the current frame, so without this the
  /// block decision for such an attack is already too late — see
  /// `decide_block_types`.
  bool lookahead_attack_gr0 = false;
  /// The next frame's energy relative to this one's, as an unbounded ratio.
  double lookahead_demand = 0.0;
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

  /// Mid/Side band attenuation bounds in dB (-6.0 to -36.0 dB) for NMR & rate optimization
  double cfg_ms_atten_min_db = 6.0;
  double cfg_ms_atten_max_db = 36.0;


  // --- Search effort (Config::effort, CLI --effort; see encoder.cpp for the
  // mapping). These were compile-time constants in beam_search.cpp, so
  // `effort` and its explicit overrides did nothing at all. Every value below
  // reproduces the old constant exactly at the default effort of 7.
  /// ± window of global-gain candidates around the bin-search seed.
  int cfg_gain_window = 4;
  /// Highest per-band scalefactor the search may try (capped by slen too).
  int cfg_sf_max = 5;
  /// Terminal DP states that get a real `final_bits` count per configuration.
  int cfg_rate_probes = 8;
  /// Resolution of the DP's cumulative-rate axis.
  int cfg_rate_buckets = 64;
  /// Sweeps of the greedy ±1 scalefactor polish after the DP.
  int cfg_polish_iters = 6;

  /// Adaptive joint-stereo: when true, the encoder decides per frame whether
  /// M/S coding actually compresses better than plain L/R and switches
  /// mode_ext accordingly (2 = MS on, 0 = MS off). Always-on MS wastes bits
  /// on weakly/anti-correlated material where E[S] ≈ E[M].
  bool cfg_adaptive_ms = true;
  /// Previous frame's MS decision (hysteresis anchor to stop flip-flopping
  /// at the decision boundary, which would otherwise modulate the stereo
  /// image frame-to-frame).
  bool ms_prev_frame = true;

  // --- High-Frequency Extension ---
  bool cfg_hf_extension = false;
  double cfg_hf_ext_cutoff_hz = -1.0;
  double cfg_hf_ext_obj_gain_db = 0.0;
  double cfg_hf_ext_res_gain_db = -1.0;
  double cfg_hf_ext_tilt_db_oct = 0.0;
  int hf_cutoff_sfb = NUM_SFB_LONG;

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
  /// A-weighted "did not keep" and "invented" powers, summed over the frame.
  /// The frame Fullness / Bleedless score is one ratio of these sums against
  /// `frame_w_signal_pow`, not a mean of per-granule decibels.
  double frame_w_def = 0.0;
  double frame_w_exc = 0.0;
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

  // ----- Circuit bends (Ali Modular) -----
  /// The caller's bends, written through Encoder::bends() between frames.
  EncoderBends bends{};
  /// What they remember from frame to frame.
  bend::State bend{};
  /// Config::fast_quantizer: skip beam_search, go straight to outer_loop.
  bool cfg_fast_quantizer = false;
};

} // namespace rmps
