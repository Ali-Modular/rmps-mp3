#pragma once
#include "rmps/bend.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rmps {

/// MPEG-1 Layer III decoder, the counterpart of Encoder.
///
/// It decodes what this library encodes — MPEG-1 at 32, 44.1 and 48 kHz,
/// mono, stereo and M/S joint stereo, long, start, stop and short blocks, with
/// the bit reservoir — and nothing it does not: no MPEG-2 LSF, no intensity
/// stereo (read as L/R), no mixed blocks (read as short). That is the whole of
/// what Encoder emits, so a stream round-trips through the pair exactly.
///
/// It exists because the encoder had no decoder to hand its frames to, and
/// because the decoder is the other half of the codec that can be bent — see
/// DecoderBends.
///
/// Streaming and allocation-free after construction: decode() takes any
/// number of bytes, frames them itself, and returns whatever PCM those bytes
/// completed.
class Decoder {
public:
    Decoder();
    ~Decoder();
    Decoder(const Decoder&)            = delete;
    Decoder& operator=(const Decoder&) = delete;

    /// Feed bytes of an MP3 stream. Returns the PCM of every frame they
    /// completed, interleaved, as floats where ±1 is full scale; always
    /// stereo (a mono stream is duplicated), 1152 frames per MP3 frame. The
    /// span is owned by the decoder until the next call. Bytes that do not
    /// complete a frame are kept for the next call.
    std::span<const float> decode(std::span<const std::uint8_t> bytes);

    /// Sample rate of the last frame decoded, or 0 before the first.
    int samplerate() const noexcept;

    /// Forget the stream: the reservoir, the filterbank overlap, any partial
    /// frame. Bends are kept.
    void reset();

    DecoderBends& bends() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rmps
