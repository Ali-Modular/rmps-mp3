#include "core/resample.hpp"

#include <algorithm>
#include <cmath>

namespace rmps::core {

namespace {

constexpr double PI = 3.14159265358979323846;

double sinc(double x) {
    if (std::abs(x) < 1e-12) return 1.0;
    const double t = PI * x;
    return std::sin(t) / t;
}

/// Blackman-Harris window. Chosen over Kaiser only because it needs no shape
/// parameter to justify: at 32 taps per phase its sidelobes are far under the
/// quantization noise of anything this encoder then does to the signal.
double blackman_harris(double n, double N) {
    const double t = 2.0 * PI * n / N;
    return 0.35875 - 0.48829 * std::cos(t) + 0.14128 * std::cos(2.0 * t) -
           0.01168 * std::cos(3.0 * t);
}

int gcd_int(int a, int b) {
    while (b) { const int t = a % b; a = b; b = t; }
    return a;
}

} // namespace

Resampler::Resampler(int in_rate, int out_rate, int channels)
    : channels_(std::max(1, channels)) {
    if (in_rate <= 0 || out_rate <= 0 || in_rate == out_rate) {
        l_ = m_ = 1;
        return;
    }
    const int g = gcd_int(in_rate, out_rate);
    l_ = out_rate / g;   // upsample
    m_ = in_rate / g;    // downsample

    // Prototype lowpass, designed in the L·fs_in intermediate domain. The
    // cutoff must clear *both* Nyquists — the upsampled image and the
    // decimated alias — so it is the tighter of the two. 0.90 backs it off
    // from the exact edge; the last tenth of the band is where a windowed sinc
    // is least flat and nothing here needs it.
    const int taps = l_ * TAPS_PER_PHASE;
    const double fc = 0.90 * 0.5 / static_cast<double>(std::max(l_, m_));
    proto_.resize(static_cast<std::size_t>(taps));
    const double center = (taps - 1) / 2.0;
    for (int j = 0; j < taps; ++j) {
        const double x = static_cast<double>(j) - center;
        proto_[static_cast<std::size_t>(j)] = static_cast<float>(
            2.0 * fc * sinc(2.0 * fc * x) *
            blackman_harris(static_cast<double>(j),
                            static_cast<double>(taps - 1)));
    }
    // Normalise each polyphase branch to unity DC gain. Per branch, not over
    // the whole prototype: the branches do not carry equal shares of the
    // window, so one global scale leaves a periodic ripple at the output rate.
    for (int p = 0; p < l_; ++p) {
        double sum = 0.0;
        for (int k = p; k < taps; k += l_) sum += proto_[static_cast<std::size_t>(k)];
        if (std::abs(sum) < 1e-12) continue;
        const auto inv = static_cast<float>(1.0 / sum);
        for (int k = p; k < taps; k += l_) proto_[static_cast<std::size_t>(k)] *= inv;
    }

    // History must cover every tap an emitted output can reach back to. The
    // oldest tap of output n is k_max - T + 1, and k_max can lag `in_count_-1`
    // by up to one output step (M/L input frames) because outputs are emitted
    // as soon as they are ready. Sizing it at exactly T drops those taps
    // silently, which is a lowpass that quietly stops being the filter it was
    // designed as.
    hist_frames_ = TAPS_PER_PHASE + (m_ + l_ - 1) / l_ + 2;
    hist_.assign(static_cast<std::size_t>(hist_frames_ * channels_), 0.0f);
}

void Resampler::emit_available(std::vector<std::int16_t>& out) {
    const std::size_t L = static_cast<std::size_t>(l_);
    const std::size_t M = static_cast<std::size_t>(m_);
    const std::size_t H = static_cast<std::size_t>(hist_frames_);
    for (;;) {
        const std::size_t base = out_count_ * M;
        const std::size_t k_max = base / L;
        const std::size_t phase = base % L;
        if (in_count_ == 0 || k_max > in_count_ - 1) return;
        for (int c = 0; c < channels_; ++c) {
            float acc = 0.0f;
            for (int t = 0; t < TAPS_PER_PHASE; ++t) {
                const auto ts = static_cast<std::size_t>(t);
                if (ts > k_max) break;                    // input index < 0
                const std::size_t k = k_max - ts;
                if (k + H < in_count_) break;             // older than history
                const std::size_t j = phase + ts * L;     // <= L*T - 1
                const std::size_t slot =
                    (k % H) * static_cast<std::size_t>(channels_) +
                    static_cast<std::size_t>(c);
                acc += proto_[j] * hist_[slot];
            }
            const float v = std::clamp(acc, -32768.0f, 32767.0f);
            out.push_back(static_cast<std::int16_t>(std::lround(v)));
        }
        ++out_count_;
    }
}

void Resampler::process(const std::int16_t* in, std::size_t frames,
                        std::vector<std::int16_t>& out) {
    if (passthrough()) {
        out.insert(out.end(), in,
                   in + frames * static_cast<std::size_t>(channels_));
        return;
    }
    const std::size_t H = static_cast<std::size_t>(hist_frames_);
    for (std::size_t f = 0; f < frames; ++f) {
        const std::size_t slot_base =
            (in_count_ % H) * static_cast<std::size_t>(channels_);
        for (int c = 0; c < channels_; ++c)
            hist_[slot_base + static_cast<std::size_t>(c)] =
                static_cast<float>(in[f * static_cast<std::size_t>(channels_) +
                                      static_cast<std::size_t>(c)]);
        ++in_count_;
        emit_available(out);
    }
}

void Resampler::flush(std::vector<std::int16_t>& out) {
    if (passthrough()) return;
    // Walk the last real samples all the way through the filter with silence.
    const std::size_t n = static_cast<std::size_t>(hist_frames_);
    std::vector<std::int16_t> zeros(n * static_cast<std::size_t>(channels_), 0);
    process(zeros.data(), n, out);
}

} // namespace rmps::core
