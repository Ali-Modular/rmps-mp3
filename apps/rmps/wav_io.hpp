#pragma once
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace rmps::wav {

struct WavInfo {
    int channels    = 0;     ///< source channels (1 or 2)
    int samplerate  = 0;
    std::uint32_t total_frames = 0;
    double duration_s   = 0.0;
};

/// Streaming 16-bit PCM WAV reader. Always emits interleaved stereo (mono is
/// up-mixed by duplication) so the encoder pipeline stays consistent.
class Reader {
public:
    static std::unique_ptr<Reader> open(const std::string& path);
    ~Reader();

    const WavInfo& info() const noexcept { return info_; }

    /// Read up to `frames` PCM frames per channel. Returns the actual number
    /// of frames read (0 = EOF). `out` is filled with `frames_read *
    /// stereo_channels (=2)` interleaved samples.
    std::size_t read(std::int16_t* out, std::size_t frames);

private:
    Reader();
    WavInfo info_ {};
    std::ifstream file_;
    int           src_channels_ = 0;
    std::uint32_t data_remaining_bytes_ = 0;
    std::vector<std::int16_t> scratch_;
};

/// Synthesised stereo sine (440 Hz L / 450 Hz R) source for smoke tests.
class SineSource {
public:
    SineSource(double freq_hz, int samplerate, double duration_s);

    WavInfo info() const noexcept;

    /// Same contract as Reader::read.
    std::size_t read(std::int16_t* out, std::size_t frames);

private:
    double freq_hz_;
    int    samplerate_;
    std::uint32_t total_frames_;
    std::uint32_t produced_ = 0;
};

} // namespace rmps::wav
