#include "post/hf_adaptor.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace rmps::post {

namespace {
constexpr double TRANSIENT_RATIO   = 4.0;
constexpr double BIT_PRESSURE_KNEE = 0.85;

/// Extra HF attenuation applied *on* a transient frame.
///
/// This is the wrong sign for brightness. The attenuation lands on the frame
/// the attack is in — a cymbal or hat loses its top exactly as it strikes —
/// and is then released over the following frames through the EMA below, so
/// the brightness arrives late instead. Reported audibly as the high end
/// lagging behind the music.
///
/// Note the cap it runs into: `hf_max_atten_m_db` is 0, so this only ever
/// attenuates *Side*. It is therefore not pre-echo control (pre-echo is a
/// Mid-dominated phenomenon and this would not touch it) — it is a bit-saving
/// measure that pays for itself by narrowing the stereo image at the top,
/// exactly on the attacks where the top matters most. Now 0 by default.
///
/// The eval harness cannot see this: on the synthetic testset the adaptor
/// never winds past its 0.3 dB engage threshold, so every metric reads
/// identical. On real material it engages at both 56 and 128 kbps (verified
/// by hash). This default is a reasoned change, not a measured one.
double transient_bias_db() {
    if (const char *e = std::getenv("RMPS_HF_TBIAS")) {
        const double v = std::atof(e);
        if (v >= 0.0) return v;
    }
    return 0.0;
}

/// Release rate of the attenuation, as a multiple of the attack rate.
///
/// `observe` is pure feedback: the attenuation applied to frame N comes from
/// frame N-1's statistics, smoothed. Winding *up* slowly is right — it keeps
/// the loop from chasing single frames. Winding *down* slowly is not: it is
/// what holds the high end suppressed after the passage that justified it has
/// passed. 1.0 is the symmetric EMA this replaced.
double release_ratio() {
    if (const char *e = std::getenv("RMPS_HF_RELEASE")) {
        const double v = std::atof(e);
        if (v >= 1.0) return v;
    }
    return 2.0;
}
}

HFAdaptor::HFAdaptor(double shelf_hz, double max_m, double max_s,
                     double target_sdr, double alpha)
    : shelf_hz_(shelf_hz)
    , max_atten_m_db_(max_m)
    , max_atten_s_db_(max_s)
    , target_sdr_db_(target_sdr)
    , ema_alpha_(alpha) {}

void HFAdaptor::apply(EncoderState& s) {
    double e = 0.0;
    for (int ch = 0; ch < s.wave.channels; ++ch) {
        for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
            const auto& arr = s.mdct_freq[ch][gr];
            for (int i = 0; i < GRANULE_SAMPLES; ++i) {
                const double v = static_cast<double>(arr[i]);
                e += v * v;
            }
        }
    }
    if (e < 1.0) e = 1.0;
    const double ratio = e / std::max(prev_energy_, 1.0);
    transient_   = ratio > TRANSIENT_RATIO;
    prev_energy_ = 0.7 * prev_energy_ + 0.3 * e;

    static const double t_bias = transient_bias_db();
    const double atten_db = cur_atten_db_ + (transient_ ? t_bias : 0.0);
    last_total_atten_db_  = atten_db;
    if (atten_db < 0.3) return;

    const int shelf_bin =
        std::clamp(static_cast<int>(shelf_hz_ * 576.0 * 2.0 / s.wave.samplerate),
                   0, GRANULE_SAMPLES);
    if (shelf_bin >= GRANULE_SAMPLES) return;

    const bool is_ms = (s.wave.channels == 2 && (s.mpeg.mode_ext & 2));
    const double m_db = std::min(atten_db, max_atten_m_db_);
    const double s_db = std::min(atten_db, max_atten_s_db_);
    const double gain_m = std::pow(10.0, -m_db / 20.0);
    const double gain_s = std::pow(10.0, -s_db / 20.0);

    for (int ch = 0; ch < s.wave.channels; ++ch) {
        const double g = is_ms ? (ch == 0 ? gain_m : gain_s) : gain_s;
        for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
            auto& arr = s.mdct_freq[ch][gr];
            for (int i = shelf_bin; i < GRANULE_SAMPLES; ++i) {
                arr[i] = fx::sat32(arr[i] * g);
            }
        }
    }
}

void HFAdaptor::observe(const FrameStats& fs) {
    if (fs.granules.empty()) return;
    const double sdr    = fs.mean_perc_sdr_db;
    const int    bits   = fs.total_bits;
    const int    budget = std::max(fs.total_budget, 1);
    const double usage  = static_cast<double>(bits) / budget;
    const double sdr_dev      = std::max(0.0, target_sdr_db_ - sdr);
    const double bit_pressure = std::max(0.0, usage - BIT_PRESSURE_KNEE) * 30.0;
    double target = 0.7 * sdr_dev + bit_pressure;
    target = std::clamp(target, 0.0, max_atten_s_db_);
    static const double rel = release_ratio();
    const double alpha = (target < cur_atten_db_)
                             ? std::min(1.0, ema_alpha_ * rel)
                             : ema_alpha_;
    cur_atten_db_ = alpha * target + (1.0 - alpha) * cur_atten_db_;
}

HFAdaptorState HFAdaptor::state() const noexcept {
    return {last_total_atten_db_, transient_, shelf_hz_};
}

} // namespace rmps::post
