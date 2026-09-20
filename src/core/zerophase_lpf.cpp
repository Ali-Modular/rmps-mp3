#include "core/zerophase_lpf.hpp"

#include <algorithm>
#include <cmath>

namespace rmps::core {

namespace {
constexpr double PI = 3.14159265358979323846;
}

ZeroPhaseLowpass::ZeroPhaseLowpass(int samplerate, double cutoff_hz,
                                   int channels, int order)
    : channels_(std::max(1, channels)) {
    order = std::clamp(order + (order & 1), 2, 8);
    // A digital Butterworth as a cascade of RBJ lowpass biquads: each biquad
    // is the bilinear transform with the frequency pre-warped at the cutoff,
    // and with the analog prototype's pole-pair Qs the cascade is the exact
    // digital Butterworth — flat passband, -3 dB per pass at `cutoff_hz`.
    const double w0 = 2.0 * PI * cutoff_hz / samplerate;
    const double cw = std::cos(w0), sw = std::sin(w0);
    for (int k = 0; k < order / 2; ++k) {
        const double theta = PI * (2.0 * k + 1.0) / (2.0 * order);
        const double q = 1.0 / (2.0 * std::sin(theta));
        const double alpha = sw / (2.0 * q);
        const double a0 = 1.0 + alpha;
        Biquad b{};
        b.b0 = (1.0 - cw) / 2.0 / a0;
        b.b1 = (1.0 - cw) / a0;
        b.b2 = (1.0 - cw) / 2.0 / a0;
        b.a1 = -2.0 * cw / a0;
        b.a2 = (1.0 - alpha) / a0;
        sections_.push_back(b);
    }
    fwd_.assign(static_cast<std::size_t>(channels_),
                std::vector<State>(sections_.size()));
    held_.assign(static_cast<std::size_t>(channels_), {});
}

void ZeroPhaseLowpass::process(const std::int16_t* in, std::size_t frames,
                               std::vector<std::int16_t>& out) {
    const std::size_t C = static_cast<std::size_t>(channels_);
    for (std::size_t c = 0; c < C; ++c) {
        auto& h = held_[c];
        h.reserve(h.size() + frames);
        auto& st = fwd_[c];
        for (std::size_t f = 0; f < frames; ++f) {
            // Transposed direct form II, one section after another.
            double x = static_cast<double>(in[f * C + c]);
            for (std::size_t k = 0; k < sections_.size(); ++k) {
                const Biquad& b = sections_[k];
                State& z = st[k];
                const double y = b.b0 * x + z.z1;
                z.z1 = b.b1 * x - b.a1 * y + z.z2;
                z.z2 = b.b2 * x - b.a2 * y;
                x = y;
            }
            h.push_back(x);
        }
    }
    in_frames_ += frames;
    while (held_[0].size() >= static_cast<std::size_t>(BLOCK + MARGIN))
        release(static_cast<std::size_t>(BLOCK), out);
}

void ZeroPhaseLowpass::release(std::size_t n_frames,
                               std::vector<std::int16_t>& out) {
    const std::size_t C = static_cast<std::size_t>(channels_);
    const std::size_t len = held_[0].size();
    n_frames = std::min(n_frames, len);
    // Never release more than the input had: the flush's zero padding exists
    // only to settle the filter and must not become output.
    n_frames = std::min(n_frames, in_frames_ - out_frames_);
    std::vector<double> back(len);
    std::vector<std::vector<double>> released(C);
    for (std::size_t c = 0; c < C; ++c) {
        // Backward pass over everything held, newest first, from zero state.
        std::vector<State> st(sections_.size());
        const auto& h = held_[c];
        for (std::size_t i = len; i-- > 0;) {
            double x = h[i];
            for (std::size_t k = 0; k < sections_.size(); ++k) {
                const Biquad& b = sections_[k];
                State& z = st[k];
                const double y = b.b0 * x + z.z1;
                z.z1 = b.b1 * x - b.a1 * y + z.z2;
                z.z2 = b.b2 * x - b.a2 * y;
                x = y;
            }
            back[i] = x;
        }
        released[c].assign(back.begin(),
                           back.begin() + static_cast<std::ptrdiff_t>(n_frames));
    }
    const std::size_t base = out.size();
    out.resize(base + n_frames * C);
    for (std::size_t f = 0; f < n_frames; ++f)
        for (std::size_t c = 0; c < C; ++c) {
            const double v = std::clamp(released[c][f], -32768.0, 32767.0);
            out[base + f * C + c] = static_cast<std::int16_t>(std::lround(v));
        }
    for (std::size_t c = 0; c < C; ++c)
        held_[c].erase(held_[c].begin(),
                       held_[c].begin() + static_cast<std::ptrdiff_t>(
                                              std::min(n_frames, held_[c].size())));
    out_frames_ += n_frames;
}

void ZeroPhaseLowpass::flush(std::vector<std::int16_t>& out) {
    // Run the forward pass on MARGIN zeros so the last real samples are
    // followed by settled context, then release everything that is real.
    const std::size_t C = static_cast<std::size_t>(channels_);
    const std::size_t real_in = in_frames_;
    std::vector<std::int16_t> zeros(static_cast<std::size_t>(MARGIN) * C, 0);
    process(zeros.data(), static_cast<std::size_t>(MARGIN), out);
    in_frames_ = real_in;                        // the padding is not input
    while (out_frames_ < in_frames_ && !held_[0].empty())
        release(in_frames_ - out_frames_, out);
}

} // namespace rmps::core
