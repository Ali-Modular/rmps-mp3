#pragma once
#include "post/perceptual.hpp"
#include "core/side_info.hpp"   // GRANULE_SAMPLES
#include <array>

namespace rmps {
struct EncoderState;

namespace post {

/// Per-SFB masking thresholds for one (ch, gr), expressed in the same
/// linear power units as `sum |xr[k]|^2` over the band. The beam_search
/// cost function compares the candidate quantization band noise D_b to
/// `thr[b]` to compute per-band NMR.
struct MaskingThresholds {
    /// Per-SFB masking threshold (linear power, MDCT² scale).
    std::array<double, NUM_SFB_LONG_> thr {};
    /// Per-SFB raw signal power (cached so beam_search doesn't recompute).
    std::array<double, NUM_SFB_LONG_> e_band {};
    /// Per-SFB tonality coefficient in [0, 1]: 0 = noise-like, 1 = tonal.
    std::array<double, NUM_SFB_LONG_> tonality {};
};

/// 3D masking model parameters. Built from Config + bitrate context.
struct MaskingParams {
    /// Hybrid weight: cost = α·NMR + (1-α)·SDR. 1.0 = pure NMR,
    /// 0.0 = pure A-weighted SDR (legacy behaviour).
    double alpha = 0.3;
    /// RDO Lagrange multiplier for bit cost in beam_search.
    /// Effective `J = D + λ_R · R`.
    double lambda_R = 0.0;
    /// Per-band post-mask decay coefficient. 1.0 = no decay (full
    /// post-mask carries forward), 0.0 = ignore previous granule.
    double post_mask_decay = 0.6;
    /// Attack-from-silence pre-mask penalty: multiplier applied to thr
    /// in granules flagged by the transient gate. < 1 lowers the
    /// threshold (more bits required). Set very low — the silent portion
    /// of the granule provides essentially zero psychoacoustic masking,
    /// so the threshold should approach the absolute threshold of
    /// hearing, not the band-averaged signal level.
    double pre_mask_attack_from_silence = 0.02;
    /// Same, for ordinary transients (mid-music attack).
    double pre_mask_transient = 0.25;
    /// Blend weight for the inter-frame (predictability) tonality estimate
    /// vs. the single-frame SFM estimate, used once temporal history is
    /// available: tonality = w·pred + (1-w)·SFM. The predictability measure
    /// (ISO Psy-2 style "chaos") is the more reliable discriminator, so it
    /// is weighted higher. Before history exists the SFM estimate is used
    /// alone.
    double inter_frame_tonality_weight = 0.6;

    /// Waveform preservation regularization strength (0..1.0).
    double waveform_reg = 0.05;

    /// Low-frequency strictness. Lowers the masking threshold of bands at or
    /// below `lf_strict_cutoff_hz` by `lf_strict_db` dB, which forces the
    /// beam search to demand lower quantization noise (≈ more bits) there.
    /// Applied AFTER the ATH floor — i.e. the model may demand precision
    /// below the absolute threshold of hearing — to clean up the bass at the
    /// cost of bits. 0 dB = disabled. The effect tapers off smoothly over
    /// ~1 Bark above the cutoff to avoid an audible band-edge discontinuity.
    double lf_strict_db        = 10.0;
    double lf_strict_cutoff_hz = 1000.0;

    // --- Pre-echo SFB-dependent penalty ---
    // High frequencies expose pre-echo more audibly because:
    //   (a) HF has less simultaneous masking from musical content
    //   (b) The ear's temporal resolution is finer at HF
    // These multipliers scale the base pre_mask penalty per SFB region.
    /// SFB 0–10: base penalty (2.0x)
    double pre_echo_hf_penalty_lo = 2.0;
    /// SFB 11–16: mid-upper penalty
    double pre_echo_hf_penalty_mid = 1.8;
    /// SFB 17–20: HF penalty (strongest)
    double pre_echo_hf_penalty_hi = 3.0;
    /// Exponent for the energy-contrast scaling. The pre-mask penalty is
    /// multiplied by (energy_contrast)^pre_echo_contrast_exponent so
    /// sharper transients get stronger suppression. 0.0 = ignore contrast.
    double pre_echo_contrast_exponent = 0.3;
};

/// Per-channel temporal state for the masking model. One per channel,
/// owned by EncoderState. The previous granule's threshold per SFB
/// is the only persistent input needed for post-masking.
struct MaskingHistory {
    std::array<double, NUM_SFB_LONG_> prev_thr {};
    bool has_prev = false;

    /// Per-bin magnitude history for the inter-frame predictability
    /// (unpredictability / "chaos") tonality estimate. `mag_m1` is the
    /// previous granule (t-1), `mag_m2` the one before (t-2), advanced in
    /// (gr, ch) processing order so the predictor is strictly causal.
    std::array<double, GRANULE_SAMPLES> mag_m1 {};
    std::array<double, GRANULE_SAMPLES> mag_m2 {};
    bool has_hist1 = false;   ///< mag_m1 valid (≥1 past granule seen)
    bool has_hist2 = false;   ///< mag_m2 valid (≥2 past granules seen)
};

/// Compute the 3D masking thresholds for (ch, gr). Must be called after
/// MDCT is filled and before beam_search runs for that granule. Updates
/// `history` for the next call.
void compute_masking_thresholds(const EncoderState& s, int ch, int gr,
                                const MaskingParams& params,
                                MaskingHistory& history,
                                MaskingThresholds& out);

/// Convenience: pick params from the encoder's runtime config + state.
MaskingParams make_masking_params(const EncoderState& s);

} // namespace post
} // namespace rmps
