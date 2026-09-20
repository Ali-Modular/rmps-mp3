#pragma once
/// Shared psychoacoustic primitives.
///
/// The long-block (masking.cpp) and short-block (masking_short.cpp) models
/// describe the same ear with different time/frequency resolutions, so every
/// piece of actual psychoacoustics — the Bark map, the SPL calibration, the
/// absolute threshold of hearing, the spreading integrals, the tone/noise
/// offset — lives here and is used by both. Only the band geometry and the
/// temporal structure differ between the two callers.

#include "core/scale.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace rmps::post::psy {

// --- Bark / frequency geometry --------------------------------------------

/// Approximate Bark scale (Traunmüller 1990). Clamped at 0: the closed form
/// evaluates to -0.53 at DC, which would otherwise make the lowest band
/// spuriously wide on the Bark axis (and every spreading integral over it
/// wrong by that much).
inline double bark_of_hz(double f_hz) {
    return std::max(0.0, 26.81 * f_hz / (1960.0 + f_hz) - 0.53);
}

// --- Level calibration (MDCT-fullscale based) ------------------------------
//
// The masking model and the ATH need to share a single power<->dB-SPL map so
// that the level-dependent spreading (which needs the masker's SPL) and the
// absolute threshold of hearing (defined in dB SPL) are mutually consistent.
//
// `e_band` is sum |xr[k]|^2 over the band, in 32-bit-integer MDCT units. A
// loud band sits around 1e14 in these units; we anchor that to ~90 dB SPL,
// giving the affine map below. The offset is the one tuning knob; everything
// downstream (ATH floor, level-dependent slope) follows from it coherently.
/// Calibrated at the reference MDCT scale (`scale::REF_SHIFT`). The live
/// value tracks the headroom shift, so changing that moves the domain without
/// moving where the ATH sits inside it.
inline constexpr double SPL_OFFSET_DB_AT_REF = -50.0;

inline double spl_offset_db() {
    return SPL_OFFSET_DB_AT_REF - scale::mdct_scale_db();
}

inline double power_to_spl(double e_band) {
    return 10.0 * std::log10(std::max(e_band, 1.0)) + spl_offset_db();
}
inline double spl_to_power(double spl_db) {
    return std::pow(10.0, (spl_db - spl_offset_db()) / 10.0);
}

// --- Absolute threshold of hearing (Terhardt), dB SPL ----------------------
//
// ATH(f) ≈ 3.64·(f/1000)^-0.8 - 6.5·exp(-0.6·(f/1000-3.3)²) + 1e-3·(f/1000)^4.
// The quartic term makes the curve climb steeply above ~16 kHz; clamp it so
// it stays finite for the highest bands.
inline double ath_db_at(double f_hz) {
    const double k = f_hz / 1000.0;
    if (k <= 0.0) return 120.0;
    const double db = 3.64 * std::pow(k, -0.8)
                    - 6.5  * std::exp(-0.6 * (k - 3.3) * (k - 3.3))
                    + 1e-3 * std::pow(k, 4.0);
    return std::clamp(db, -20.0, 120.0);
}

/// Minimum ATH over [lo_hz, hi_hz]. The most audible frequency in a band —
/// not its centre — governs whether quantization noise there is audible.
inline double ath_min_db_over(double lo_hz, double hi_hz) {
    constexpr int N_SAMPLES = 8;
    double m = 1e9;
    for (int i = 0; i <= N_SAMPLES; ++i) {
        const double f = lo_hz + (hi_hz - lo_hz) * (i / double(N_SAMPLES));
        m = std::min(m, ath_db_at(std::max(f, 1.0)));
    }
    return m;
}

// --- Spreading function (frequency axis), level-dependent ------------------
//
// The spread is modelled as a two-sided exponential *density* on the Bark
// axis, h(dz) = C·exp(-a_d·|dz|) below the masker and C·exp(-a_u·dz) above,
// normalised so ∫h(dz)d(dz) = 1. Writing it as a normalised density (rather
// than a point-to-point gain that happens to be 1 at dz = 0) is what makes
// the model energy-conserving: Σ_b S(b,m) ≈ 1, so a masker distributes its
// power across the spectrum instead of being amplified or diluted by the
// arbitrary number of bands that happen to be nearby.
//
// Decay rates come from the slopes in dB/Bark: a = σ·ln(10)/10. The downward
// skirt (toward LF) is steep and roughly level-independent (~27 dB/Bark). The
// upward skirt is the level-dependent one: it shallows out at high masker
// SPL, so a loud band spreads much further up the spectrum than a quiet one.
inline constexpr double LN10_OVER_10 = 0.23025850929940458;
inline constexpr double LOWER_SLOPE_DB_PER_BARK = 27.0;

/// Upper slope magnitude in dB/Bark: 24 + 0.23/f_kHz - 0.2·L (classic MPEG /
/// Bosi formulation), clamped to a sane range. Larger L (louder masker) →
/// smaller slope → more upward spread.
inline double upper_slope_db_per_bark(double f_masker_khz, double masker_spl_db) {
    const double s = 24.0 + 0.23 / std::max(f_masker_khz, 0.1)
                          - 0.2 * masker_spl_db;
    return std::clamp(s, 6.0, 30.0);
}

/// 1 - e^{-x}, accurate for small x.
inline double one_minus_exp(double x) { return -std::expm1(-x); }

/// Normalisation constant of the two-sided exponential density.
inline double spread_norm(double a_u, double a_d) {
    return 1.0 / (1.0 / a_u + 1.0 / a_d);
}

/// Fraction of a masker band's power that the spreading density deposits in a
/// *non-overlapping* target band.
///
/// Both bands are treated as uniform power densities over their Bark extent
/// (masker width `w_m`, target width `w_b`, edge-to-edge gap `gap`), and the
/// density is integrated over both. This is the correction a centre-to-centre
/// point model gets badly wrong: two adjacent 1.5-Bark bands have touching
/// edges but centres 1.5 Bark apart, so the point model under-estimates the
/// spread between them by 10 dB or more. Integrating removes that error
/// without needing a fine sub-band grid.
///
///   S = C·(1-e^{-a·w_m})(1-e^{-a·w_b})·e^{-a·gap} / (w_m·a²)
///
/// with `a` the decay rate on the side the target sits.
inline double spread_fraction(double C, double a, double w_m, double w_b,
                              double gap) {
    return C * one_minus_exp(a * w_m) * one_minus_exp(a * w_b)
             * std::exp(-a * gap) / (w_m * a * a);
}

/// Fraction a band deposits in *itself*: the same double integral taken over
/// the band's own square, split at the diagonal so each half uses the decay
/// rate for its direction.
inline double spread_self(double C, double a_u, double a_d, double w) {
    const auto F = [w](double a) {
        const double x = a * w;
        if (x < 0.05) {
            // Series expansion — the closed form below cancels catastrophically
            // as x → 0.
            return w * w * (0.5 - x / 6.0 + x * x / 24.0);
        }
        const double e = std::exp(-x);
        return w * one_minus_exp(x) / a - (1.0 - e * (1.0 + x)) / (a * a);
    };
    return C * (F(a_u) + F(a_d)) / w;
}

// --- Tonality / tone-noise offset ------------------------------------------

/// Tonality from the Spectral Flatness Measure.
/// SFM = exp(mean(log P_k)) / mean(P_k). 1.0 = perfectly flat (noise), 0.0 =
/// single tone. Mapped to [0,1] via Johnston's -log10(SFM)/6 relation.
inline double tonality_from_sfm(const std::int32_t* xrabs, int n_bins) {
    if (n_bins < 2) return 0.5;
    double sum_log = 0.0;
    double sum_lin = 0.0;
    int    counted = 0;
    for (int k = 0; k < n_bins; ++k) {
        const double v = static_cast<double>(xrabs[k]);
        const double p = v * v + 1.0;  // floor to keep log finite
        sum_log += std::log(p);
        sum_lin += p;
        ++counted;
    }
    const double gm = std::exp(sum_log / counted);
    const double am = sum_lin / counted;
    const double sfm = (am > 0.0) ? std::min(gm / am, 1.0) : 1.0;
    const double dB = -10.0 * std::log10(std::max(sfm, 1e-6));
    return std::clamp(dB / 60.0, 0.0, 1.0);   // 0 dB flat → 0; 60 dB → 1
}

/// Signal-to-mask offset in dB from tonality. Tone-masking-noise (TMN) is
/// roughly 14.5 dB below the masker; noise-masking-tone (NMT) about 5.5 dB.
/// Interpolate linearly on the tonality axis.
inline double smr_db(double tonality_alpha) {
    constexpr double SMR_NOISE_DB = -5.5;
    constexpr double SMR_TONE_DB  = -14.5;
    return SMR_NOISE_DB + tonality_alpha * (SMR_TONE_DB - SMR_NOISE_DB);
}

/// Effective masker level for the level-dependent slope.
///
/// The slope wants the SPL of the *masker*, but a band sum spans a width that
/// varies from well under to well over one critical band. Summed noise power
/// therefore reads "louder" purely because a band is wide, tilting the slope
/// with band geometry rather than with the signal. For noise-like content we
/// normalise to a per-critical-band density; for tonal content the energy
/// really is concentrated in one component, so the raw band sum is the right
/// level. Interpolate on tonality.
inline double effective_masker_spl(double e_band, double width_bark,
                                   double tonality) {
    const double w_crit = std::max(width_bark, 1.0);
    const double e_eff = tonality * e_band
                       + (1.0 - tonality) * (e_band / w_crit);
    return power_to_spl(e_eff);
}

// --- Harmonic geometry ------------------------------------------------------

/// Refine a peak at integer bin `i` to a fractional bin by fitting a parabola
/// through its two neighbours.
///
/// This matters far more than it looks. Harmonic positions are predicted as
/// k · f0, so any rounding error in f0 is multiplied by the partial number: a
/// 220 Hz fundamental sits at bin 5.75 of the 576-line MDCT, and rounding it
/// to 6 puts the 39th partial at bin 234 instead of 224 — ten bins, 378 Hz,
/// against protection windows that are ±2 bins wide. Every consumer of the
/// harmonic map (the valley zeroing, the perceptual weight discount, the
/// masking-threshold boost) therefore acts on the wrong bins from about the
/// 4th partial upward, and the real partials get treated as valley noise.
inline double refine_peak_bin(const double *mag, int n, int i) {
  if (i <= 0 || i + 1 >= n) return static_cast<double>(i);
  const double a = mag[i - 1], b = mag[i], c = mag[i + 1];
  const double den = a - 2.0 * b + c;
  if (den == 0.0) return static_cast<double>(i);
  const double d = std::clamp(0.5 * (a - c) / den, -0.5, 0.5);
  return static_cast<double>(i) + d;
}

// --- Binaural masking level difference (Side channel) -----------------------

/// Threshold offset in dB for the *Side* channel of an M/S pair.
///
/// The rest of this header models a single ear. Side-channel content is
/// interaurally out of phase, and the binaural system detects an out-of-phase
/// signal well below the monaural threshold — the binaural masking level
/// difference (BMLD). It is largest at low frequencies, ~15 dB below 500 Hz,
/// and has essentially vanished by 1.5 kHz, where interaural phase stops being
/// resolvable. So the Side threshold must be *lowered* there: noise the
/// monaural model calls masked is audible as an unstable, wandering image.
///
/// Above that region the sign flips. Fine phase detail is not resolved, the
/// image is carried by envelope rather than phase, and a few dB of extra
/// headroom in Side is genuinely inaudible — which is what makes discarding
/// high-band Side content legitimate rather than merely cheap.
inline double side_threshold_offset_db(double f_hz) {
    constexpr double BMLD_MAX_DB   = 15.0;   // full binaural advantage, ≤500 Hz
    constexpr double BMLD_FLAT_HZ  = 500.0;
    constexpr double BMLD_ZERO_HZ  = 1500.0; // phase locking gone
    constexpr double HF_RELIEF_DB  = 3.0;    // headroom well above the BMLD band
    constexpr double HF_RELIEF_HZ  = 6000.0;
    if (f_hz <= BMLD_FLAT_HZ) return -BMLD_MAX_DB;
    if (f_hz < BMLD_ZERO_HZ) {
        const double t = (f_hz - BMLD_FLAT_HZ) / (BMLD_ZERO_HZ - BMLD_FLAT_HZ);
        return -BMLD_MAX_DB * (1.0 - t);
    }
    if (f_hz >= HF_RELIEF_HZ) return HF_RELIEF_DB;
    const double t = (f_hz - BMLD_ZERO_HZ) / (HF_RELIEF_HZ - BMLD_ZERO_HZ);
    return HF_RELIEF_DB * t;
}

// --- Low-frequency strictness ----------------------------------------------

/// Multiplier (≤ 1) applied to the masking threshold so the encoder demands
/// lower noise (more bits) at low frequencies. Full `strict_db` below the
/// cutoff, tapering linearly to no-op over ~1 Bark above it to avoid a hard
/// band-edge step.
inline double lf_strict_factor(double center_hz, double cutoff_hz,
                               double strict_db) {
    if (strict_db <= 0.0) return 1.0;
    const double z     = bark_of_hz(center_hz);
    const double z_cut = bark_of_hz(cutoff_hz);
    double t;                                  // 1 = full effect, 0 = none
    if      (z <= z_cut)       t = 1.0;
    else if (z >= z_cut + 1.0) t = 0.0;
    else                       t = 1.0 - (z - z_cut);
    return std::pow(10.0, (-strict_db * t) / 10.0);
}

// --- Perceptual entropy -----------------------------------------------------

/// Σ_b n_b · log2(1 + √(e_b / thr_b)). The square root turns the power ratio
/// into an amplitude ratio, so the term counts (roughly) the bits per line
/// needed to push quantization noise under the mask; a band already below its
/// threshold contributes ~0. This is the demand signal the bit allocator
/// wants: unlike energy or SDR it already folds in every masking effect.
inline double perceptual_entropy_of(const double* e_band, const double* thr,
                                    const int* n_bins, int n_bands) {
    double pe = 0.0;
    for (int b = 0; b < n_bands; ++b) {
        const double t = std::max(thr[b], 1.0);
        const double e = std::max(e_band[b], 0.0);
        if (e <= t) continue;
        pe += static_cast<double>(n_bins[b]) * std::log2(1.0 + std::sqrt(e / t));
    }
    return pe;
}

} // namespace rmps::post::psy
