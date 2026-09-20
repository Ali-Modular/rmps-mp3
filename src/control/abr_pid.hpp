#pragma once
#include "rmps/stats.hpp"
#include <optional>

namespace rmps {
struct EncoderState;

namespace control {

/// Snap kbps to the closest valid MPEG bitrate for the encoder's version and
/// recompute slot timing. Returns the snapped value.
int switch_bitrate(EncoderState &s, double new_kbps);

class AbrController {
public:
  struct Options {
    double target_kbps = 128.0;
    std::optional<double> target_sdr_db{};
    /// Quality-targeted VBR: hold the measured SDR at `target_sdr_db` and let
    /// the bitrate go wherever that needs, instead of holding an average rate
    /// with SDR as a bias. `target_kbps` then only seeds the first frame.
    bool quality_mode = false;
    double sdr_weight = 0.7;
    double kp = 0.8, ki = 0.24, kd = 0.1;
    double min_kbps = 0.0; ///< 0 -> target * 0.5
    double max_kbps = 0.0; ///< 0 -> target * 1.6
    double integral_leak = 0.95;
    double smoothing = 0.2;
    int mpeg_version = 3;
  };

  explicit AbrController(Options opt);

  /// Feed back a completed frame's measured bits + duration + SDR.
  /// `frame_level_db` is the frame's RMS in dBFS; quality mode tilts the SDR
  /// setpoint with it (see QUIET_TILT).
  /// `overflow_bits` is what the reservoir discarded as stuffing this frame
  /// because the bank was full: produced, paid for, and useless.
  void observe(int frame_bits, double frame_seconds,
               std::optional<double> frame_sdr_db,
               std::optional<double> frame_level_db = std::nullopt,
               int overflow_bits = 0);

  /// Feed-forward demand for the frame about to be coded: its perceptual
  /// entropy in bits (see control/demand.hpp) and its duration.
  ///
  /// Quality mode sets the rate from this directly, and uses the PID only as a
  /// slow multiplicative trim that pulls the long-run SDR onto target. A pure
  /// feedback loop cannot do this job: it learns a frame was starved only
  /// after coding it, so every fast level change — a release, a decay tail,
  /// the gap between phrases — is mis-rated for as long as the loop takes to
  /// catch up. Measured on real material, quiet frames came out 2.3 dB worse
  /// than loud ones at 128 kbps and 4.2 dB worse at 64 despite the loop
  /// holding a constant SDR setpoint. `pe` already knows what the frame needs
  /// before a bit is spent, so there is nothing to catch up to.
  ///
  /// Must be called before `set_transient_hint`, which still overrides.
  /// Ignored outside quality mode.
  void set_demand(double pe_bits, double frame_seconds) noexcept;

  /// Lookahead hint from the transient gate, supplied AFTER the PCM
  /// has been analysed but BEFORE next_target_kbps() is consumed.
  /// Overrides the PID's natural decision for the upcoming frame:
  ///   attack_from_silence → jump to max_kbps (kill pre-echo)
  ///   any_transient       → bias halfway toward max_kbps
  ///   all_silent          → bias halfway toward min_kbps
  /// Long-term average is restored by the PID's integral after the
  /// burst is observed.
  void set_transient_hint(bool attack_from_silence, bool any_transient,
                          bool all_silent) noexcept;

  /// Next-frame kbps target (last decision, after any transient hint).
  double next_target_kbps() const noexcept { return last_decision_kbps_; }

  /// The encode's running average rate — its operating point, as distinct
  /// from any one frame's decision. Drives `EncoderState::policy_kbps`.
  double operating_kbps() const noexcept { return last_avg_kbps_; }

  AbrState state() const noexcept;

private:
  Options o_;
  double integral_ = 0.0;
  double prev_err_ = 0.0;
  double cum_bits_ = 0.0;
  double cum_seconds_ = 0.0;
  double cum_bits_local_ = 0.0;
  double cum_seconds_local_ = 0.0;
  double last_decision_kbps_ = 0.0;
  double last_avg_kbps_ = 0.0;
  double last_err_ = 0.0;
  double last_u_ = 0.0;
  /// Quality mode: smoothed PID output in dB, applied multiplicatively on top
  /// of the feed-forward demand. Carries the loop's whole memory, so the
  /// decision itself no longer has to be smoothed — which is what kept the
  /// feed-forward path instantaneous.
  double trim_db_ = 0.0;
  double last_ff_kbps_ = 0.0;   ///< diagnostics only
  /// Rate ceiling learned from reservoir overflow: while the bank keeps
  /// overflowing, the content has shown it cannot use more than about this
  /// much. <= 0 means no ceiling. See `observe`.
  double ovf_ceiling_kbps_ = 0.0;
  /// Whether the last decision was held down by `ovf_ceiling_kbps_`, so the
  /// integrator does not wind up against a rate the content cannot use.
  bool ovf_binding_ = false;
  /// The decision as it was before the ceiling cut it, so a transient hint can
  /// release the ceiling and restore it within the same frame.
  double pre_ceiling_decision_kbps_ = 0.0;
  /// Lowest legal bitrate for this MPEG version (32 for MPEG-1), independent of
  /// the ABR band `o_.min_kbps`. The overflow ceiling may go down to this.
  double legal_min_kbps_ = 32.0;
  long frames_ = 0;
  bool is_silent_frame_ = false;
};

} // namespace control
} // namespace rmps
