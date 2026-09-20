#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace rmps::core {

/// Zero-phase (forward-backward) Butterworth lowpass for interleaved 16-bit
/// PCM, applied to the input before anything else sees it.
///
/// Forward-backward filtering cancels the IIR's phase exactly, so the filter
/// adds no group delay and no phase distortion, and it does not smear an
/// attack forward in time the way a causal IIR does. It does *not* remove
/// ringing: any band-limiting response rings near its cutoff (Gibbs), and
/// running it backward as well makes that ringing symmetric — the pre-ringing
/// is as large as the post-ringing. Measured on a unit step, order 4: 4.6 %
/// overshoot on each side at 44.1 kHz / 20 kHz, 7.6 % at 48 kHz / 20 kHz, 6.0 %
/// at 32 kHz / 14 kHz, each lasting under 1 ms and sitting at the cutoff
/// frequency. Real material has little energy that close to 20 kHz, so the
/// ringing it actually produces is far smaller than a step's.
///
/// Streaming. The backward pass needs samples that have not arrived yet, so
/// forward-filtered output is held until `MARGIN` further samples exist; the
/// backward pass then runs from the newest sample back and only the oldest
/// block — where its zero-state start has long decayed — is released. The
/// released stream is sample-aligned with the input (no latency in the
/// signal), it simply arrives in blocks, and `flush` releases the rest with
/// exactly the input's length.
class ZeroPhaseLowpass {
public:
    /// Samples of settled backward-pass context required before a block is
    /// released. The order-4 impulse response falls under -60 dB within about
    /// 1 ms (~45 samples at 44.1 kHz); 2048 leaves its start-up error at the
    /// rounding floor.
    static constexpr int MARGIN = 2048;
    static constexpr int BLOCK = 4096;

    /// `order` is per pass (even, 2..8); the combined response is its square.
    ZeroPhaseLowpass(int samplerate, double cutoff_hz, int channels, int order = 4);

    void process(const std::int16_t* in, std::size_t frames,
                 std::vector<std::int16_t>& out);
    void flush(std::vector<std::int16_t>& out);

private:
    struct Biquad {
        double b0, b1, b2, a1, a2;
    };
    struct State {
        double z1 = 0.0, z2 = 0.0;
    };
    void release(std::size_t n_frames, std::vector<std::int16_t>& out);

    int channels_ = 1;
    std::vector<Biquad> sections_;
    /// Forward-pass state, per channel per section, carried across calls.
    std::vector<std::vector<State>> fwd_;
    /// Forward-filtered samples not yet released, one deque-like vector per
    /// channel (front = oldest).
    std::vector<std::vector<double>> held_;
    std::size_t in_frames_ = 0;
    std::size_t out_frames_ = 0;
};

} // namespace rmps::core
