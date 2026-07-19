#pragma once
#include "rmps/config.hpp"
#include "rmps/stats.hpp"
#include "rmps/version.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rmps {

/// PCM input format. Always interleaved 16-bit signed.
struct PcmChunk {
    const std::int16_t* data;
    std::size_t         num_samples;   ///< total interleaved samples
};

class Encoder {
public:
    /// Construct from a public config. Returns null on bad samplerate/bitrate.
    static std::unique_ptr<Encoder> create(const Config& cfg);

    ~Encoder();
    Encoder(const Encoder&)            = delete;
    Encoder& operator=(const Encoder&) = delete;

    /// Number of PCM frames the encoder consumes per call to encode().
    std::size_t samples_per_pass() const noexcept;

    /// Encode a chunk of exactly `samples_per_pass()` PCM frames *per channel*
    /// (interleaved). Returns the encoded MP3 bytes for one MP3 frame. The
    /// returned span is owned by the encoder until the next call.
    std::span<const std::uint8_t> encode(PcmChunk pcm);

    /// Flush any pending bytes (call once at end of stream).
    std::span<const std::uint8_t> flush();

    /// Finalized Xing ("Xing" for ABR/VBR, "Info" for CBR) tag frame with
    /// frame count, byte count and a 100-point seek TOC. A same-sized
    /// placeholder was already emitted at the very start of the stream;
    /// after flush(), overwrite the first vbr_tag().size() bytes of a
    /// seekable output with this span. Unseekable sinks (stdout) can skip
    /// the rewrite and keep the placeholder. The span is owned by the
    /// encoder; empty if no audio frames were produced.
    std::span<const std::uint8_t> vbr_tag();

    /// Latest frame's stats. Valid after every encode() call.
    const FrameStats& last_frame_stats() const noexcept;

    /// Live ABR controller state, if ABR is active; nullptr otherwise.
    const AbrState* abr_state() const noexcept;

    /// Mutate the runtime CBR/ABR target. For ABR, sets PID setpoint.
    void set_target_kbps(int kbps);

private:
    Encoder();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rmps
