#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace rmps::app {

struct AudioInfo {
    int channels = 2;            ///< always 2: sources are delivered as stereo
    int samplerate = 0;
    int source_channels = 0;     ///< what the file actually has
    std::uint64_t total_frames = 0;
    double duration_s = 0.0;
    std::string format;          ///< "FLAC", "OGG/Opus", "WAV/PCM_24", ...
};

/// A PCM source delivering interleaved 16-bit stereo.
///
/// Stereo always, as the WAV reader always did, so every input reaches the
/// encoder in the shape it was tuned on: mono is duplicated into both
/// channels, and more than two channels are folded down (front L/R plus
/// half of the centre, the remaining channels split evenly), with the result
/// scaled so a full-scale input cannot clip.
class AudioSource {
public:
    virtual ~AudioSource() = default;
    virtual const AudioInfo& info() const noexcept = 0;
    /// Read up to `frames` frames into `out` (frames * 2 samples). Returns the
    /// number read; 0 at end of stream.
    virtual std::size_t read(std::int16_t* out, std::size_t frames) = 0;
};

/// Open any supported audio file (UTF-8 path). With libsndfile: WAV, AIFF,
/// FLAC, Ogg Vorbis, Ogg Opus and the other formats it reads; without it,
/// WAV only. Throws std::runtime_error with a readable message on failure.
std::unique_ptr<AudioSource> open_audio(const std::string& utf8_path);

/// Synthesised stereo sine for smoke tests (L = freq, R = freq + 10 Hz).
std::unique_ptr<AudioSource> make_sine(double freq_hz, int samplerate,
                                       double duration_s);

/// Human-readable list of what `open_audio` accepts, for help text and the
/// GUI's file filter.
const char* supported_formats();

} // namespace rmps::app
