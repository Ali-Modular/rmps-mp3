#include "control/abr_pid.hpp"
#include "core/encoder_state.hpp"
#include "core/tables.hpp"

#include <algorithm>
#include <cmath>

namespace rmps::control {

namespace {

int find_nearest_bitrate_index(double kbps, int version, int& snapped) {
    int best_idx = 1;
    int best_diff = INT_MAX;
    int best_kbps = 0;
    for (int i = 1; i < tables::NUM_BITRATES - 1; ++i) {
        const int b = tables::bitrates[i][version];
        if (b < 0) continue;
        const int diff = std::abs(static_cast<int>(std::lround(kbps)) - b);
        if (diff < best_diff) {
            best_diff = diff;
            best_idx  = i;
            best_kbps = b;
        }
    }
    snapped = best_kbps;
    return best_idx;
}

} // namespace

int switch_bitrate(EncoderState& s, double new_kbps) {
    int snapped = 0;
    const int idx = find_nearest_bitrate_index(new_kbps, s.mpeg.version, snapped);
    if (idx == s.mpeg.bitrate_index && snapped == s.mpeg.bitr) return snapped;

    s.mpeg.bitrate_index = idx;
    s.mpeg.bitr          = snapped;
    const double avg_slots =
        (s.mpeg.granules_per_frame * 576.0 / s.wave.samplerate)
        * (1000.0 * snapped / s.mpeg.bits_per_slot);
    s.mpeg.whole_slots_per_frame = static_cast<int>(avg_slots);
    s.mpeg.frac_slots_per_frame  = avg_slots - s.mpeg.whole_slots_per_frame;
    return snapped;
}

AbrController::AbrController(Options opt) : o_(std::move(opt)) {
    const int version = o_.mpeg_version;
    int lo = INT_MAX, hi = 0;
    for (int i = 1; i < tables::NUM_BITRATES - 1; ++i) {
        const int b = tables::bitrates[i][version];
        if (b < 0) continue;
        lo = std::min(lo, b);
        hi = std::max(hi, b);
    }
    if (o_.min_kbps <= 0.0) {
        const double target_min = o_.target_kbps - 12.0;
        int best_b = hi;
        for (int i = 1; i < tables::NUM_BITRATES - 1; ++i) {
            const int b = tables::bitrates[i][version];
            if (b < 0) continue;
            if (b >= target_min && b < best_b) {
                best_b = b;
            }
        }
        o_.min_kbps = static_cast<double>(best_b);
    }
    if (o_.max_kbps <= 0.0) {
        const double target_max = o_.target_kbps + 48.0;
        int best_b = hi;
        for (int i = 1; i < tables::NUM_BITRATES - 1; ++i) {
            const int b = tables::bitrates[i][version];
            if (b < 0) continue;
            if (b >= target_max && b < best_b) {
                best_b = b;
                break;
            }
        }
        o_.max_kbps = static_cast<double>(best_b);
    }
    o_.min_kbps = std::max(o_.min_kbps, static_cast<double>(lo));
    o_.max_kbps = std::min(o_.max_kbps, static_cast<double>(hi));
    last_decision_kbps_ = o_.target_kbps;
    last_avg_kbps_      = o_.target_kbps;
}

void AbrController::observe(int frame_bits, double frame_seconds,
                            std::optional<double> frame_sdr_db) {
    if (frame_seconds <= 0.0) return;
    // Warm-start the running average with a virtual history at the target, so
    // the cumulative mean does not wind up from the first (often max-bitrate
    // cold-start) frame and drag the PID through a large convergence ramp.
    // That startup ramp is the dominant source of HF-cutoff "swirl" under ABR
    // (the steady-state cutoff jitter already matches CBR). ~25 frames ≈ 0.65 s.
    if (frames_ == 0) {
        constexpr double WARM_FRAMES = 25.0;
        cum_seconds_ = WARM_FRAMES * frame_seconds;
        cum_bits_    = WARM_FRAMES * o_.target_kbps * 1000.0 * frame_seconds;
        cum_seconds_local_ = WARM_FRAMES * frame_seconds;
        cum_bits_local_    = WARM_FRAMES * o_.target_kbps * 1000.0 * frame_seconds;
    }
    cum_bits_    += frame_bits;
    cum_seconds_ += frame_seconds;
    
    // Update local average with a decay factor (~20 frames window)
    constexpr double LOCAL_DECAY = 0.95;
    cum_bits_local_    = LOCAL_DECAY * cum_bits_local_ + frame_bits;
    cum_seconds_local_ = LOCAL_DECAY * cum_seconds_local_ + frame_seconds;
    
    ++frames_;

    const double avg_local = (cum_bits_local_ / cum_seconds_local_) / 1000.0;
    const double avg_global = (cum_bits_ / cum_seconds_) / 1000.0;
    last_avg_kbps_ = avg_global;

    // Local error for Proportional and Derivative terms (incorporating quality/SDR target)
    double err_local = o_.target_kbps - avg_local;
    if (o_.target_sdr_db && frame_sdr_db) {
        err_local += o_.sdr_weight * (*o_.target_sdr_db - *frame_sdr_db);
    }
    
    // Global error for Integral term (WITHOUT quality/SDR bias, ensuring perfect long-term convergence)
    const double err_global = o_.target_kbps - avg_global;
    
    // Use a longer leak memory (0.995) for the global rate error integration.
    // Freeze the integrator during silent frames to prevent windup.
    const double integral_leak_to_use = std::max(0.995, o_.integral_leak);
    if (!is_silent_frame_) {
        integral_ = integral_leak_to_use * integral_ + err_global;
    } else {
        integral_ = integral_leak_to_use * integral_;
    }

    const double deriv_local = err_local - prev_err_;
    prev_err_ = err_local;
    last_err_ = err_local;

    const double u = o_.kp * err_local + o_.ki * integral_ + o_.kd * deriv_local;
    last_u_ = u;

    const double target = is_silent_frame_ ? 32.0 : (o_.target_kbps + u);
    last_decision_kbps_ =
        o_.smoothing * target + (1.0 - o_.smoothing) * last_decision_kbps_;
    const double effective_min = is_silent_frame_ ? 32.0 : o_.min_kbps;
    last_decision_kbps_ = std::clamp(last_decision_kbps_, effective_min, o_.max_kbps);
    // Deadband toward the target: snap small deviations exactly to the target
    // so the steady-state bitrate index (hence the HF cutoff) stops toggling
    // between adjacent steps frame-to-frame. Reduced to 3% to allow proper ABR tracking.
    constexpr double DEADBAND_FRAC = 0.03;
    if (std::abs(last_decision_kbps_ - o_.target_kbps)
            < DEADBAND_FRAC * o_.target_kbps)
        last_decision_kbps_ = o_.target_kbps;
}

void AbrController::set_transient_hint(bool attack_from_silence,
                                       bool any_transient,
                                       bool all_silent) noexcept {
    is_silent_frame_ = all_silent;
    if (attack_from_silence) {
        // Hard force: this is the worst pre-echo case, spend everything
        // we can on it. The PID's average tracking will pull subsequent
        // frames down to compensate.
        last_decision_kbps_ = o_.max_kbps;
    } else if (any_transient) {
        // Bias halfway toward the upper clamp from the current decision.
        last_decision_kbps_ = std::min(
            o_.max_kbps,
            last_decision_kbps_ + 0.5 * (o_.max_kbps - last_decision_kbps_));
    } else if (all_silent) {
        // Conserve bits during true silence. The PID would do this
        // eventually via the average; the hint just makes it immediate.
        last_decision_kbps_ = std::max(
            32.0,
            last_decision_kbps_ - 0.5 * (last_decision_kbps_ - 32.0));
    }
}

AbrState AbrController::state() const noexcept {
    AbrState st {};
    st.target_kbps   = o_.target_kbps;
    st.avg_kbps      = last_avg_kbps_;
    st.decision_kbps = last_decision_kbps_;
    st.err           = last_err_;
    st.u             = last_u_;
    st.integral      = integral_;
    st.min_kbps      = o_.min_kbps;
    st.max_kbps      = o_.max_kbps;
    st.frames        = frames_;
    return st;
}

} // namespace rmps::control
