#pragma once
#include "post/masking.hpp"       // MaskingParams / MaskingHistory
#include "core/side_info.hpp"     // NUM_WINDOWS
#include <array>

namespace rmps {
struct EncoderState;

namespace post {

/// Scalefactor bands the short-block model covers. The bitstream transmits
/// scalefactors for 12 of them (NUM_SFB_SHORT); the 13th is coded with an
/// implicit scalefactor of 0 but still carries signal, so the psychoacoustic
/// model includes it.
inline constexpr int NUM_SFB_SHORT_PSY = 13;

/// Per-(window, sfb) masking thresholds for one short granule.
///
/// Short granules were previously the one place with no psychoacoustic model
/// at all: iteration_loop skipped them (the long model's band layout does not
/// map onto the reordered short spectrum) and the short quantizer fell back on
/// a hand-tuned "weight the bass 12x" rule. That left the encoder without a
/// mask exactly where pre-echo is decided.
struct ShortMaskingThresholds {
    std::array<std::array<double, NUM_SFB_SHORT_PSY>, NUM_WINDOWS> thr {};
    std::array<std::array<double, NUM_SFB_SHORT_PSY>, NUM_WINDOWS> e_band {};
    /// Perceptual entropy over all three windows (bits), on the same scale as
    /// MaskingThresholds::pe so the bit allocator can use them interchangeably.
    double pe = 0.0;
    bool valid = false;
};

/// Per-channel temporal state for the short model. Only the last window's
/// threshold needs to survive to the next granule.
struct ShortMaskingHistory {
    std::array<double, NUM_SFB_SHORT_PSY> prev_thr {};
    bool has_prev = false;
};

/// Compute per-(window, sfb) thresholds for a short granule.
///
/// Reads the reordered short spectrum from `s.mdct_freq[ch][gr]` (transmission
/// order: for each sfb, for each window, a contiguous run of lines) and models:
///   • Bark-domain spreading within each 4.3 ms window, using the same
///     energy-conserving integrals as the long model;
///   • the absolute threshold of hearing;
///   • forward masking from the previous window (and, across the granule
///     boundary, from the previous granule's last window);
///   • backward masking from the *next* window — deliberately weak, since
///     backward masking dies within ~2 ms and the windows are 4.3 ms apart.
///     This is what makes the window before an attack demand bits: it gets
///     almost no help from the attack that follows it, which is precisely the
///     pre-echo condition.
void compute_short_masking_thresholds(const EncoderState& s, int ch, int gr,
                                      const MaskingParams& params,
                                      ShortMaskingHistory& history,
                                      ShortMaskingThresholds& out);

/// Fold a short granule's thresholds back into the long model's history.
///
/// Without this the long model's post-mask carry stalls across every short
/// granule: the granule after an attack inherits a threshold computed before
/// it, which is both stale and (because the attack raised levels) far too low.
void project_short_to_long(const EncoderState& s,
                           const ShortMaskingThresholds& st,
                           MaskingHistory& long_history);

} // namespace post
} // namespace rmps
