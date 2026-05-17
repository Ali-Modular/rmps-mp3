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
    if (o_.min_kbps <= 0.0) o_.min_kbps = o_.target_kbps * 0.5;
    if (o_.max_kbps <= 0.0) o_.max_kbps = o_.target_kbps * 1.6;
    o_.min_kbps = std::max(o_.min_kbps, static_cast<double>(lo));
    o_.max_kbps = std::min(o_.max_kbps, static_cast<double>(hi));
    last_decision_kbps_ = o_.target_kbps;
    last_avg_kbps_      = o_.target_kbps;
}

void AbrController::observe(int frame_bits, double frame_seconds,
                            std::optional<double> frame_sdr_db) {
    if (frame_seconds <= 0.0) return;
    cum_bits_    += frame_bits;
    cum_seconds_ += frame_seconds;
    ++frames_;

    const double avg = (cum_bits_ / cum_seconds_) / 1000.0;
    last_avg_kbps_ = avg;
    double err = o_.target_kbps - avg;
    if (o_.target_sdr_db && frame_sdr_db) {
        err += o_.sdr_weight * (*o_.target_sdr_db - *frame_sdr_db);
    }
    integral_ = o_.integral_leak * integral_ + err;
    const double deriv = err - prev_err_;
    prev_err_ = err;
    last_err_ = err;

    const double u = o_.kp * err + o_.ki * integral_ + o_.kd * deriv;
    last_u_ = u;

    const double target = o_.target_kbps + u;
    last_decision_kbps_ =
        o_.smoothing * target + (1.0 - o_.smoothing) * last_decision_kbps_;
    last_decision_kbps_ = std::clamp(last_decision_kbps_, o_.min_kbps, o_.max_kbps);
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
