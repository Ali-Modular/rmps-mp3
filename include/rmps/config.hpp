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

  /// Sample rate the stream is *coded* at, when it should differ from the
  /// input's. 0 = choose automatically from the operating bitrate; any other
  /// value is used as-is (set it equal to `samplerate` to disable resampling).
  ///
  /// At low bitrates 44.1 kHz is the wrong operating point and always was.
  /// MPEG-1 stereo side info is a fixed 256 bits per frame no matter how few
  /// bits the frame has, so at 48 kbps it is 256 of 1254 bits — 20.4 % of the
  /// stream spent before a single spectral coefficient is coded. Coding at
  /// 32 kHz leaves the side info the same per frame but makes the frame 36 ms
  /// instead of 26.1 ms, which is 14.8 % instead of 20.4 %.
  ///
  /// The bandwidth it costs is bandwidth already being discarded: the adaptive
  /// Mid lowpass floor is 10 kHz at 48 kbps and 11 kHz at 64 (see
  /// `post/lowpass.cpp`), both well under a 32 kHz stream's 16 kHz Nyquist. In
  /// exchange all 576 MDCT lines land under 16 kHz rather than under 22.05,
  /// so every scalefactor band is narrower in Hz for the same line count.
  ///
  /// 32 kHz is deliberate rather than the more aggressive 22.05: it keeps the
  /// stream MPEG-1, whose scalefactor coding is what this encoder implements.
  /// MPEG-2 LSF would halve the side info again and is the better answer at
  /// 48 kbps and below, but its `scalefac_compress` is a different scheme
  /// (four partitions, no scfsi, no preflag) that the bitstream writer and the
  /// two quantizer front-ends do not implement.
  int coded_samplerate = 0;

  /// Zero-phase (forward-backward) Butterworth lowpass on the input, before
  /// resampling and analysis: 20 kHz when the stream is coded at 44.1 or
  /// 48 kHz, 14 kHz when coded at 32 kHz. No phase shift and no group delay;
  /// its ringing is symmetric and sits at the cutoff (see
  /// `core/zerophase_lpf.hpp`). RMPS_PRE_LPF=0 disables it.
  bool prefilter_lowpass = true;

  // --- Adaptive features ---
  /// Optional ABR target in kbps. nullopt => CBR at `bitrate`.
  std::optional<int> abr_target_kbps{};
  int abr_min_kbps = 0; ///< 0 = use abr_target - 12 (rounded up)
  int abr_max_kbps = 0; ///< 0 = use abr_target + 48 (rounded up)
  /// Optional perceptual SDR floor (dB) for ABR; nullopt disables steering.
  std::optional<double> target_sdr_db{90.0};

  /// Quality-targeted VBR. When set, this is the *only* thing the caller
  /// specifies: the encoder holds the measured perceptual SDR at this value
  /// and moves the bitrate wherever that requires, across the full legal
  /// range. The CLI exposes it as `-q 1..100`, mapped linearly onto 10-32 dB;
  /// the library takes the dB directly. Every other quality decision — effort, noise shaping, the
  /// masking-model weights, the search widths — is the encoder's to make, so
  /// none of them is exposed. Overrides `bitrate` / `abr_target_kbps`.
  std::optional<double> quality_sdr_db{};
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
  /// HF-to-LF energy ratio an attack must clear to be coded with short
  /// blocks. `decide_block_types` requires `transient && hf_transient`, so
  /// this single number decides whether short blocks are used at all.
  ///
  /// At the old 1.5 they effectively were not. Census on real material
  /// (Witchqueen, 36k channel-granules at 128 kbps): 2326 transients
  /// detected, 37 passed the HF test — 1.6 % — giving 0.21 % short granules.
  /// mixed_music was worse: 87 transients, *zero* HF-qualified, zero short
  /// blocks. The ratio is first-difference energy over boxcar-averaged
  /// energy, and on real music, which has bass under everything, it rarely
  /// reaches 1.5 even for a hi-hat.
  ///
  /// 0.4 triples the qualifying rate (37 -> 183 granules, 0.98 % short) and
  /// measured 0 better / 0 worse on every metric at 64/128/192 — the testset
  /// does not exercise this path, so that is "no regression", not "no
  /// effect". 0.2 reaches 2.20 % short but costs 2 cells of segSNR_p05.
  /// Sweepable with RMPS_SB_HF_THR.
  ///
  /// Now 0.2, chosen against the artefact rather than against the harness.
  ///
  /// The user-reported symptom was a click added just before a hit at 48-96
  /// kbps. Localised by measuring, in the decoded file, samples that exceed the
  /// *source's own local peak* — with the source measured against itself as a
  /// control, since ordinary program material puts 1.5-3.6 % of its samples
  /// over that line by crest factor alone. The excess over the control sits
  /// overwhelmingly in the 5 ms immediately *before* an onset (avicii 36 % at
  /// 64 kbps and 41 % at 96; shishamo 76 % at 96), i.e. it is pre-echo, and
  /// short blocks are the mechanism that exists to stop it. On real material
  /// only 1.1 % of granules were getting one.
  ///
  /// Sweeping the threshold on avicii at 64 kbps (short-granule share, then
  /// spike samples in that 5 ms window):
  ///
  ///     0.4   1.1 %   65      <- was
  ///     0.2   1.5 %   40      <- knee
  ///     0.1   2.6 %   42
  ///     0.05  5.0 %   42
  ///
  /// so 0.2 removes 38 % of them and nothing below it buys more.
  ///
  /// The harness only sees the cost, and only on one clip: `broadband_noise`
  /// loses 0.30-0.38 dB of segSNR at 96/128 and 1.72 dB of segSNR_p05 at 128,
  /// with every other clip and metric unmoved (`pre_echo_db` included, since
  /// `transient_train` does not exercise the HF classifier). That clip is
  /// stationary noise: short blocks there are a detector false positive on
  /// material that does not occur in music, and paying for it to fix an
  /// audible click on program material is the right way round. This is a case
  /// where the synthetic battery and the real measurement disagree and the
  /// real one wins — see tools/README.md on what the battery is for.
  double short_block_hf_threshold = 0.2;   ///< was 1.5, then 0.4

  // --- Quality knobs ---
  /// Search effort in [0..10], exposed on the CLI as `-e/--effort`. Default 2.
  /// Scales the quantizer search linearly (effort 7 lands on the search's
  /// original constants): scalefactor range 1 at 0-2, 2 at 3, 3 at 4, 4 at
  /// 5-6, 5 at 7+; gain window, rate probes/buckets and polish passes grow
  /// with it. The scalefactor range is the lever for keeping quiet bands next
  /// to loud ones — at the default of 1 a band can sit at most one step finer
  /// than the granule's gain. Measured with the range alone widened to 3
  /// (RMPS_SF_MAX=3, effort otherwise 2): hole rate 9 better / 4 worse with
  /// very large wins (hf_hiss_decay@48k 41.3 -> 9.6 %), segSNR 0 / 14,
  /// ~1.4x encode time. RMPS_SF_MAX still overrides the range alone.
  int effort = 2;
  /// Explicit overrides for beam search knobs. If negative, they are
  /// auto-derived from the `effort` level.
  int beam_width = -1;
  int gain_window = -1; ///< ±N around bin-search seed
  int sf_max_candidate = -1;
  double noise_shaping_alpha = 0.8; ///< 0..0.95 (killer-sample fit 2026-07-20)
  double hf_max_atten_m_db = 0.0;   ///< HFAdaptor cap (M)
  double hf_max_atten_s_db = 0.0;   ///< HFAdaptor cap (S)

  /// Mid/Side band attenuation bounds for bit-budget & NMR RDO optimization (-6
  /// dB to -36 dB)
  double ms_atten_min_db = 0.0; ///< Min attenuation step (dB)
  double ms_atten_max_db = 0.0; ///< Max attenuation limit (dB)

  // --- 3D masking model (beam-search cost function) ---
  /// Mix between NMR (perceptual, mask-aware) and A-weighted SDR:
  ///   cost = mask_alpha · NMR + (1 - mask_alpha) · SDR
  /// 0.0 = legacy SDR (safe default during model bring-up).
  /// 0.3 = NMR-assisted, SDR-dominant (recommended starter).
  /// 1.0 = pure NMR.
  double mask_alpha = 0.7; ///< killer-sample fit 2026-07-20 (was 0.3)
  /// Override for the per-band RDO Lagrange multiplier λ_R in the
  /// beam-search cost. Negative = auto-derive from bitrate (default).
  double mask_lambda_R = -1.0;

  /// Low-frequency strictness: lower the masking threshold of bands at or
  /// below `lf_strict_cutoff_hz` by this many dB, forcing the beam search
  /// to spend more bits keeping the bass clean. Applied *after* the ATH
  /// floor, so it may demand precision below absolute audibility. Default
  /// 10 dB (strong, bass-favouring); set to 0.0 to disable.
  double lf_strict_db = 6.0; ///< killer-sample fit 2026-07-20 (was 6.0)
  /// Upper edge (Hz) of the LF strictness region. The effect is full below
  /// this and tapers off over ~1 Bark above it. Default 1 kHz.
  double lf_strict_cutoff_hz = 1400.0;

  /// Input-side pre-attenuation in dB applied to PCM samples before
  /// any analysis. The codec has a structural ~1 dB loudness shift
  /// (cause not yet localised — see investigation pending). -1.0 dB
  /// is the default compensation; pass 0.0 to disable and 1:1 the
  /// raw pipeline behaviour.
  double pre_gain_db = -1.0;

  /// Per-band perceptual weight bias applied to SFBs 17–20 (≈7–16 kHz
  /// at 44.1k). Multiplies the A-weighted band weight; values > 1
  /// force the beam-search to spend more bits preserving HF noise
  /// (hi-hat sizzle, cymbal shimmer), at the cost of LF/mid SDR.
  /// Default 2.5 (strong HF preservation). Higher values bias toward HF.
  double hf_weight_boost = 1.5; ///< killer-sample fit 2026-07-20 (was 2.5)

  /// Boost cap in dB (0.0 to 24.0) for the transparent zero fill
  /// ("black-spot boost") in HF noise-like bands (≈5 kHz+). The fill is
  /// compressive (level-dependent): a zeroed coefficient near the dead-zone
  /// boundary is kept at ≈0 dB lift, a quieter one is lifted by half its
  /// depth in dB, saturating at this cap — so low-level detail is recovered
  /// without uniform-gain over-fill. Realised as |ix|=1 promotions whose
  /// density follows the curve; total energy is capped at the band's actual
  /// deficit and disabled on transient/pre-echo granules, so it closes
  /// spectral holes without ever reading as an HF boost. 0 disables.
  /// Default 4.0.
  double hf_zero_boost_db = 1.0; ///< killer-sample fit 2026-07-20 (was 4.0)

  /// Waveform preservation regularization strength in [0..1.0].
  /// Penalizes raw distortion independently of psychoacoustic masking
  /// thresholds, preserving overall waveform shape. Default 0.05.
  /// Strength of the source-difference leg in the beam cost
  /// (`waveform_reg · w · N(D/e_ref)`, added outside the NMR/level blend).
  ///
  /// This is the *only* live lever on "stay close to the source". The two
  /// obvious ones are not: `mask_alpha` and the noise-fill weight are both
  /// *blends* between two legs that are each monotone in the coding error, so
  /// re-weighting them leaves the search's ranking almost untouched — alpha
  /// measured bit-identical from 0.70 all the way to 0.00, and the fill
  /// weight inert at a 67 % cut. An additive leg with its own multiplier does
  /// change the relative weighting between bands, which is what allocation
  /// actually responds to.
  ///
  /// Measured on the testset at 64/128/320 (segSNR better/worse, hole rate
  /// mean Δ): 1e3 → 2/0, -0.19 · 1e4 → 3/0, -1.24 · 1e6 → 5/0, -2.09.
  /// It is the segSNR-vs-black-spot trade in its purest form, so 1e3 takes
  /// the knee: a third of the available segSNR for a sixth of the hole cost.
  /// Raise toward 1e4 to push fidelity harder at the cost of more holes.
  double waveform_reg = 1e3; ///< was 64.0 (killer-sample fit 2026-07-20)

  // --- PID gains override ---
  std::optional<double> abr_kp{};
  std::optional<double> abr_ki{};
  std::optional<double> abr_kd{};

  // --- High-Frequency Extension (Bandwidth Extension) ---
  /// Enable online high-frequency bandwidth extension. When enabled, missing
  /// high frequencies are synthesized from lower bands and used per subband
  /// if the synthesized HF yields lower rate-distortion cost.
  bool hf_extension = false;
  double hf_ext_cutoff_hz = -1.0; ///< <0 = auto-detect cutoff
  double hf_ext_obj_gain_db = 0.0;
  double hf_ext_res_gain_db = -1.0;
  double hf_ext_tilt_db_oct = 0.0;

  // --- Real time (Ali Modular) ---
  /// Quantize every long granule with the rate loop alone (`outer_loop`, a
  /// global-gain bisection) instead of the beam search. The beam search is
  /// what this encoder's quality is made of, and it is also ~95 % of its
  /// time: measured at 128 kbps, 2.2x real time with it and 36x without. An
  /// encoder that has to keep up with an audio callback cannot afford it.
  bool fast_quantizer = false;
  /// Bank bits across frames. Off, every frame carries its own main data and
  /// main_data_begin stays 0, which removes the frames of output lag the
  /// reservoir adds - the same switch as RMPS_NO_RESV, as a setting.
  bool bit_reservoir = true;
};

} // namespace rmps
