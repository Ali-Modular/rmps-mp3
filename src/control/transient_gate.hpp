#pragma once
#include <array>
#include <cstdint>

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

private:
  Decision decisions_[2]{};
  int transient_count_ = 0;
  /// Energy of the last analysed sub-window. Negative = never analysed
  /// (cold start sentinel; distinct from "silent", which is 1.0 after the
  /// floor clamp inside the measurer).
  double last_window_energy_ = -1.0;
};

} // namespace control
} // namespace rmps
