#pragma once
#include <cstdint>
#include <optional>

namespace rmps {

enum class Mode : std::uint8_t {
  Stereo = 0,
  JointStereo = 1,
  DualChannel = 2,
  Mono = 3,
};

enum class Emphasis : std::uint8_t {
  None = 0,
  Mu5015 = 1,
  CITT = 3,
};

enum class MpegVersion : std::uint8_t {
  Mpeg25 = 0,
  MpegII = 2,
  MpegI = 3,
};

/// Public encoder configuration.
struct Config {
  int channels = 2;       ///< 1 (mono) or 2 (stereo)
  int samplerate = 44100; ///< Hz; valid: 8/11.025/12/16/22.05/24/32/44.1/48k
  int bitrate = 128;      ///< kbps (CBR default, ABR initial)
  Mode mode = Mode::JointStereo; ///< forced to JointStereo for MS when stereo
  Emphasis emph = Emphasis::None;
  bool copyright = false;
  bool original = true;

  // --- Adaptive features ---
  /// Optional ABR target in kbps. nullopt => CBR at `bitrate`.
  std::optional<int> abr_target_kbps{};
  int abr_min_kbps = 0;  ///< 0 = use abr_target - 12 (rounded up)
  int abr_max_kbps = 0;  ///< 0 = use abr_target + 48 (rounded up)
  /// Optional perceptual SDR floor (dB) for ABR; nullopt disables steering.
  std::optional<double> target_sdr_db{90.0};
  /// Enable adaptive joint-stereo (M/S vs L/R decision per frame).
  bool adaptive_joint_stereo = true;
  /// Enable short-block window switching (block_type 2) on transients. When
  /// false the encoder stays long-block-only (legacy behaviour).
  bool block_switching = true;
  /// Minimum ratio of an attack's high-frequency energy (first difference) to
  /// its bass / low-mid energy (16-tap boxcar low-pass) for it to switch to a
  /// short block. Only HF-dominated transients (hi-hats, snare snaps, cymbals)
  /// clear this; broadband bass-heavy hits (kicks, bass-note onsets, snares
  /// with body) stay on long blocks so their bass keeps full frequency
  /// resolution — short blocks otherwise turn the sustained bass into audible
  /// quant noise at the attack. Higher = more conservative (fewer short
  /// blocks). Default 5.0.
  double short_block_hf_threshold = 5.0;

  // --- Quality knobs ---
  int beam_width = 128;
  int gain_window = 32; ///< ±N around bin-search seed
  int sf_max_candidate = 12;
  double noise_shaping_alpha = 0.25; ///< 0..0.95
  double hf_max_atten_m_db = 0.0;    ///< HFAdaptor cap (M)
  double hf_max_atten_s_db = 1.0;    ///< HFAdaptor cap (S)

  // --- 3D masking model (beam-search cost function) ---
  /// Mix between NMR (perceptual, mask-aware) and A-weighted SDR:
  ///   cost = mask_alpha · NMR + (1 - mask_alpha) · SDR
  /// 0.0 = legacy SDR (safe default during model bring-up).
  /// 0.3 = NMR-assisted, SDR-dominant (recommended starter).
  /// 1.0 = pure NMR.
  double mask_alpha = 0.0;
  /// Override for the per-band RDO Lagrange multiplier λ_R in the
  /// beam-search cost. Negative = auto-derive from bitrate (default).
  double mask_lambda_R = -1.0;

  /// Low-frequency strictness: lower the masking threshold of bands at or
  /// below `lf_strict_cutoff_hz` by this many dB, forcing the beam search
  /// to spend more bits keeping the bass clean. Applied *after* the ATH
  /// floor, so it may demand precision below absolute audibility. Default
  /// 10 dB (strong, bass-favouring); set to 0.0 to disable.
  double lf_strict_db = 12.0;
  /// Upper edge (Hz) of the LF strictness region. The effect is full below
  /// this and tapers off over ~1 Bark above it. Default 1 kHz.
  double lf_strict_cutoff_hz = 1000.0;

  /// Input-side pre-attenuation in dB applied to PCM samples before
  /// any analysis. The codec has a structural ~2 dB loudness shift
  /// (cause not yet localised — see investigation pending). -2.0 dB
  /// is the default compensation; pass 0.0 to disable and 1:1 the
  /// raw pipeline behaviour.
  double pre_gain_db = -1.4;

  /// Per-band perceptual weight bias applied to SFBs 17–20 (≈7–16 kHz
  /// at 44.1k). Multiplies the A-weighted band weight; values > 1
  /// force the beam-search to spend more bits preserving HF noise
  /// (hi-hat sizzle, cymbal shimmer), at the cost of LF/mid SDR.
  /// Default 2.5 (strong HF preservation). Higher values bias toward HF.
  double hf_weight_boost = 2.5;

  /// Waveform preservation regularization strength in [0..1.0].
  /// Penalizes raw distortion independently of psychoacoustic masking thresholds,
  /// preserving overall waveform shape. Default 0.05.
  double waveform_reg = 0.05;

  // --- PID gains override ---
  std::optional<double> abr_kp{};
  std::optional<double> abr_ki{};
  std::optional<double> abr_kd{};
};

} // namespace rmps
