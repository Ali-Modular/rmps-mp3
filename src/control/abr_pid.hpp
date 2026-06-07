#pragma once
#include "rmps/stats.hpp"
#include <optional>

namespace rmps {
struct EncoderState;

namespace control {

/// Snap kbps to the closest valid MPEG bitrate for the encoder's version and
/// recompute slot timing. Returns the snapped value.
int switch_bitrate(EncoderState& s, double new_kbps);

class AbrController {
public:
    struct Options {
        double target_kbps   = 128.0;
        std::optional<double> target_sdr_db {};
        double sdr_weight    = 0.5;
        double kp = 0.5, ki = 0.12, kd = 0.04;
        double min_kbps      = 0.0;     ///< 0 -> target * 0.5
        double max_kbps      = 0.0;     ///< 0 -> target * 1.6
        double integral_leak = 0.97;
        double smoothing     = 0.4;
        int    mpeg_version  = 3;
    };

    explicit AbrController(Options opt);

    /// Feed back a completed frame's measured bits + duration + SDR.
    void observe(int frame_bits, double frame_seconds,
                 std::optional<double> frame_sdr_db);

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

    AbrState state() const noexcept;

private:
    Options o_;
    double integral_      = 0.0;
    double prev_err_      = 0.0;
    double cum_bits_      = 0.0;
    double cum_seconds_   = 0.0;
    double last_decision_kbps_ = 0.0;
    double last_avg_kbps_      = 0.0;
    double last_err_           = 0.0;
    double last_u_             = 0.0;
    long   frames_             = 0;
};

}}
