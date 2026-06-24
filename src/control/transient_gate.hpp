#pragma once
#include <array>
#include <cstdint>
#include <cstddef>

namespace rmps {
struct EncoderState;

namespace control {

/// Per-granule transient detector and pre-echo suppressor.
///
/// Long-block MP3 cannot localise quantization noise in time: the noise of
/// a transient (kick attack, hi-hat tick, etc.) smears across the entire 13 ms
/// MDCT window and appears as audible *pre-echo* in the silence before the
/// transient.
///
/// Without short blocks, the best mitigation is to (1) cut the HF MDCT
/// coefficients of the transient granule so the wide-band noise that would
/// otherwise leak has less energy to leak with, and (2) inflate the per-granule
/// bit budget so the quantizer can keep the (now narrower) spectrum closer to
/// the source.
///
/// This gate scans the *raw* PCM that's about to feed MDCT, runs a peak/RMS
/// jump test per 192-sample sub-window, and publishes per-granule flags +
/// suggested HF cutoff. The integrating side (iteration_loop) applies the
/// cutoff and biases `pe` for the reservoir.
class TransientGate {
public:
  static constexpr int SUBS_PER_GR = 3;

  struct Decision {
    bool transient = false;
    bool pre_window = false; ///< this granule precedes a transient
    /// Set when this granule's attack lands after a sub-window of true
    /// silence. Pre-echo in this case smears into an audibly empty
    /// region, so the integrating side must apply a stronger HF cut and
    /// a bigger bit budget than for a transient mid-music.
    bool attack_from_silence = false;
    /// All sub-windows of this granule are below the silence floor.
    /// The integrating side can starve such granules so the reservoir
    /// banks up for the next attack.
    bool silent = false;
    double cutoff_hz = 20000.0;  ///< HF cutoff (post-MDCT zero)
    double peak_rms_ratio = 0.0; ///< measured strength (for UI)

    /// Per-sub-window energy (3 windows × 192 samples). The masking
    /// model and beam_search use these to estimate how much of the
    /// granule is "quiet before the attack" vs "attack itself".
    std::array<double, SUBS_PER_GR> sub_energies{0.0, 0.0, 0.0};

    /// Index (0–2) of the sub-window where the attack fires, or -1 if
    /// no attack in this granule. Earlier index ⇒ shorter pre-echo
    /// smear within the granule; later index ⇒ longer quiet-before.
    int attack_sub_index = -1;

    /// Energy contrast: ratio of attack sub-window energy to the
    /// preceding sub-window energy. Higher ⇒ sharper transient ⇒
    /// more audible pre-echo. 1.0 when no attack.
    double energy_contrast = 1.0;

    /// True when the attack carries significant high-frequency content (a
    /// hi-hat / snare / broadband click) rather than being a low-frequency
    /// thump (kick, bass-note onset). Only HF-rich attacks justify a short
    /// block: short blocks cost bass frequency resolution (audible bass quant
    /// noise) while their pre-echo benefit only matters for HF transients.
    bool hf_transient = false;
  };

  /// Inspect PCM for the frame about to be encoded. `pcm` holds
  /// `samples_per_pass * channels` interleaved 16-bit samples (the same
  /// buffer that will be fed to MDCT). After this call `decision_for()`
  /// returns valid per-granule flags.
  void analyse_frame(const std::int16_t *pcm, std::size_t samples_per_pass,
                     int channels, int granules_per_frame);

  Decision decision_for(int gr) const noexcept;

  /// Number of granules in the last analysed frame that were flagged.
  int transient_count() const noexcept { return transient_count_; }

  /// Minimum high-frequency-to-bass energy ratio for an attack to be classed as
  /// HF-rich (and thus eligible for a short block). Higher = more conservative
  /// (fewer short blocks, safer bass). Set from Config::short_block_hf_threshold.
  void set_hf_transient_threshold(double t) noexcept { hf_transient_threshold_ = t; }

  /// Query the previous frame's last granule decision. Valid only after
  /// at least one call to analyse_frame(). Used by the encoder to
  /// retroactively mark the last granule of the previous frame as
  /// pre_window when the current frame's first granule is a transient.
  const Decision& prev_frame_last_decision() const noexcept {
    return prev_frame_last_decision_;
  }

private:
  Decision decisions_[2]{};
  int transient_count_ = 0;
  /// Energy of the last analysed sub-window. Negative = never analysed
  /// (cold start sentinel; distinct from "silent", which is 1.0 after the
  /// floor clamp inside the measurer).
  double last_window_energy_ = -1.0;

  /// Snapshot of the previous frame's last granule decision, for
  /// cross-frame pre_window propagation.
  Decision prev_frame_last_decision_{};

  /// HF-transient classification threshold (see set_hf_transient_threshold).
  double hf_transient_threshold_ = 5.0;
};

/// Translate the per-granule transient/pre-window flags currently held in
/// `s` into per-(ch, gr) block types (BLOCK_LONG/START/SHORT/STOP), enforcing
/// valid window-switching transitions across granule and frame boundaries via
/// `s.prev_block_type`. Both channels of a granule always receive the same
/// block type (required for joint-stereo M/S validity). When
/// `s.cfg_block_switching` is false every granule stays BLOCK_LONG. Must run
/// after the transient gate has populated the flags and before mdct::analyse.
void decide_block_types(EncoderState& s);

} // namespace control
} // namespace rmps
