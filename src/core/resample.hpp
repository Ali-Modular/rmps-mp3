#pragma once
#include <cstdint>
#include <vector>

namespace rmps::core {

/// Streaming rational-ratio resampler for interleaved 16-bit PCM.
///
/// Exists so the encoder can choose its *coded* sample rate independently of
/// the file's. At low bitrates 44.1 kHz is the wrong operating point: the
/// per-frame side info is a fixed 256 bits for MPEG-1 stereo however few bits
/// the frame has, so at 48 kbps it is 20.4 % of a 1254-bit frame, and the 576
/// MDCT lines are spread over a 22.05 kHz span of which the lowpass is already
/// discarding everything above 10 kHz. Coding the same material at 32 kHz
/// leaves the side info unchanged per frame but makes the frame 36 ms instead
/// of 26.1 ms — 14.8 % overhead instead of 20.4 % — and puts all 576 lines
/// under 16 kHz, so every scalefactor band is narrower in Hz.
///
/// Polyphase, so the zeros the L-fold upsampling would insert are never
/// multiplied: output n is a T-tap dot product against the input, with the tap
/// set chosen by n·M mod L.
class Resampler {
public:
    /// Taps per polyphase branch. 32 puts the transition band well inside the
    /// margin the lowpass already imposes, at 32 multiply-adds per output
    /// sample per channel.
    static constexpr int TAPS_PER_PHASE = 32;

    /// `in_rate` / `out_rate` are reduced internally; passing equal rates
    /// makes `process` a copy.
    Resampler(int in_rate, int out_rate, int channels);

    bool passthrough() const noexcept { return l_ == 1 && m_ == 1; }

    /// Resample `frames` interleaved frames and append the result to `out`.
    void process(const std::int16_t* in, std::size_t frames,
                 std::vector<std::int16_t>& out);

    /// Drain the filter's history with zero input so the tail is not lost.
    void flush(std::vector<std::int16_t>& out);

    /// Output frames produced per input frame, as a ratio, for sizing buffers.
    int up() const noexcept { return l_; }
    int down() const noexcept { return m_; }

private:
    void emit_available(std::vector<std::int16_t>& out);

    int l_ = 1;          ///< upsampling factor
    int m_ = 1;          ///< downsampling factor
    int channels_ = 1;
    std::vector<float> proto_;   ///< prototype filter, length l_ * TAPS_PER_PHASE
    /// Input history, interleaved, holding the last `hist_frames_` frames.
    std::vector<float> hist_;
    int hist_frames_ = 0;
    std::size_t in_count_ = 0;   ///< input frames consumed so far
    std::size_t out_count_ = 0;  ///< output frames produced so far
};

} // namespace rmps::core
