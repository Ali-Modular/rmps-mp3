#include "wav_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace rmps::wav {

namespace {

std::uint32_t read_le32(std::istream& s) {
    std::uint8_t b[4];
    s.read(reinterpret_cast<char*>(b), 4);
    return  std::uint32_t(b[0])
         | (std::uint32_t(b[1]) <<  8)
         | (std::uint32_t(b[2]) << 16)
         | (std::uint32_t(b[3]) << 24);
}
std::uint16_t read_le16(std::istream& s) {
    std::uint8_t b[2];
    s.read(reinterpret_cast<char*>(b), 2);
    return std::uint16_t(b[0]) | std::uint16_t(b[1] << 8);
}

/// Decode one interleaved source sample at `p` to a signed 16-bit value,
/// according to the source bit depth / float flag. Integer formats are
/// arithmetically right-shifted (high bits kept); float is scaled and clamped.
std::int16_t decode_sample(const std::uint8_t* p, int bits, bool is_float) {
    auto from_double = [](double d) -> std::int16_t {
        d = std::clamp(d, -1.0, 1.0);
        long v = std::lround(d * 32767.0);
        return static_cast<std::int16_t>(std::clamp<long>(v, -32768, 32767));
    };
    if (is_float) {
        if (bits == 64) {
            double d; std::memcpy(&d, p, 8);
            return from_double(d);
        }
        float f; std::memcpy(&f, p, 4);
        return from_double(static_cast<double>(f));
    }
    switch (bits) {
        case 8: // WAV 8-bit PCM is unsigned, centred at 128
            return static_cast<std::int16_t>((static_cast<int>(p[0]) - 128) << 8);
        case 16:
            return static_cast<std::int16_t>(
                static_cast<std::uint16_t>(p[0]) |
                static_cast<std::uint16_t>(p[1]) << 8);
        case 24: {
            // Place the 24 bits in the top of a 32-bit word, then shift down
            // arithmetically to sign-extend, and again to reach 16-bit.
            std::int32_t v = static_cast<std::int32_t>(
                static_cast<std::uint32_t>(p[0]) << 8  |
                static_cast<std::uint32_t>(p[1]) << 16 |
                static_cast<std::uint32_t>(p[2]) << 24);
            return static_cast<std::int16_t>((v >> 8) >> 8);
        }
        case 32: {
            std::int32_t v = static_cast<std::int32_t>(
                static_cast<std::uint32_t>(p[0])       |
                static_cast<std::uint32_t>(p[1]) << 8  |
                static_cast<std::uint32_t>(p[2]) << 16 |
                static_cast<std::uint32_t>(p[3]) << 24);
            return static_cast<std::int16_t>(v >> 16);
        }
    }
    return 0;
}

} // namespace

Reader::Reader() = default;
Reader::~Reader() = default;

std::unique_ptr<Reader> Reader::open(const std::string& path) {
    auto r = std::unique_ptr<Reader>(new Reader());
    r->file_.open(path, std::ios::binary);
    if (!r->file_) throw std::runtime_error("cannot open WAV: " + path);

    char hdr[12];
    r->file_.read(hdr, 12);
    if (std::memcmp(hdr,     "RIFF", 4) != 0 ||
        std::memcmp(hdr + 8, "WAVE", 4) != 0) {
        throw std::runtime_error("not a RIFF/WAVE file: " + path);
    }

    bool got_fmt = false;
    bool got_data= false;
    std::uint16_t fmt_tag = 0;
    while (r->file_ && (!got_fmt || !got_data)) {
        char chunk_id[4];
        r->file_.read(chunk_id, 4);
        if (r->file_.gcount() != 4) break;
        std::uint32_t chunk_sz = read_le32(r->file_);

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            fmt_tag                = read_le16(r->file_);
            r->src_channels_       = read_le16(r->file_);
            r->info_.samplerate    = static_cast<int>(read_le32(r->file_));
            (void)read_le32(r->file_);          // byte rate
            (void)read_le16(r->file_);          // block align
            std::uint16_t bps      = read_le16(r->file_);
            std::uint32_t consumed = 16;
            // For WAVE_FORMAT_EXTENSIBLE the real sample format lives in the
            // first two bytes of the SubFormat GUID inside the extension.
            std::uint16_t actual_fmt = fmt_tag;
            if (fmt_tag == 0xFFFE) {
                std::uint16_t cb = read_le16(r->file_); consumed += 2;
                if (cb >= 22) {
                    (void)read_le16(r->file_);        // valid bits per sample
                    (void)read_le32(r->file_);        // channel mask
                    actual_fmt = read_le16(r->file_); // SubFormat GUID lead bytes
                    consumed += 8;
                }
            }
            if (chunk_sz > consumed)
                r->file_.seekg(chunk_sz - consumed, std::ios::cur);

            constexpr std::uint16_t WAVE_PCM   = 1;
            constexpr std::uint16_t WAVE_FLOAT = 3;
            const bool is_float = (actual_fmt == WAVE_FLOAT);
            if (actual_fmt != WAVE_PCM && actual_fmt != WAVE_FLOAT)
                throw std::runtime_error("unsupported WAV format (only PCM / IEEE float)");
            if (is_float) {
                if (bps != 32 && bps != 64)
                    throw std::runtime_error("float WAV must be 32- or 64-bit");
            } else if (bps != 8 && bps != 16 && bps != 24 && bps != 32) {
                throw std::runtime_error("PCM WAV must be 8/16/24/32-bit");
            }
            if (r->src_channels_ < 1 || r->src_channels_ > 2)
                throw std::runtime_error("only mono/stereo WAV supported");
            r->src_bits_  = bps;
            r->src_float_ = is_float;
            got_fmt = true;
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            r->data_remaining_bytes_ = chunk_sz;
            got_data = true;
            break; // file cursor now at PCM data
        } else {
            r->file_.seekg(chunk_sz, std::ios::cur); // skip unknown chunk
        }
    }
    if (!got_fmt || !got_data) throw std::runtime_error("missing fmt/data chunk");

    const int src_ch = r->src_channels_;
    const std::uint32_t frame_bytes =
        static_cast<std::uint32_t>(src_ch) * (static_cast<std::uint32_t>(r->src_bits_) / 8u);
    r->info_.total_frames = r->data_remaining_bytes_ / frame_bytes;
    r->info_.channels     = 2;       // we always emit stereo
    r->info_.duration_s   = static_cast<double>(r->info_.total_frames)
                          / static_cast<double>(r->info_.samplerate);
    return r;
}

std::size_t Reader::read(std::int16_t* out, std::size_t frames) {
    if (data_remaining_bytes_ == 0) return 0;
    const std::uint32_t bytes_per_sample = static_cast<std::uint32_t>(src_bits_) / 8u;
    const std::uint32_t frame_bytes =
        static_cast<std::uint32_t>(src_channels_) * bytes_per_sample;
    const std::uint32_t max_frames  = data_remaining_bytes_ / frame_bytes;
    if (frames > max_frames) frames = max_frames;
    if (frames == 0) return 0;

    const std::size_t want_bytes = frames * frame_bytes;
    if (raw_.size() < want_bytes) raw_.resize(want_bytes);
    file_.read(reinterpret_cast<char*>(raw_.data()),
               static_cast<std::streamsize>(want_bytes));
    data_remaining_bytes_ -= static_cast<std::uint32_t>(want_bytes);

    const std::uint8_t* p = raw_.data();
    for (std::size_t i = 0; i < frames; ++i) {
        const std::int16_t l = decode_sample(p, src_bits_, src_float_);
        p += bytes_per_sample;
        std::int16_t r = l;                       // mono -> duplicate
        if (src_channels_ == 2) {
            r = decode_sample(p, src_bits_, src_float_);
            p += bytes_per_sample;
        }
        out[2*i    ] = l;
        out[2*i + 1] = r;
    }
    return frames;
}

// ---------------- Sine source ----------------

SineSource::SineSource(double freq_hz, int samplerate, double duration_s)
    : freq_hz_(freq_hz)
    , samplerate_(samplerate)
    , total_frames_(static_cast<std::uint32_t>(static_cast<double>(samplerate) * duration_s)) {}

WavInfo SineSource::info() const noexcept {
    WavInfo i;
    i.channels    = 2;
    i.samplerate  = samplerate_;
    i.total_frames= total_frames_;
    i.duration_s  = static_cast<double>(total_frames_) / static_cast<double>(samplerate_);
    return i;
}

std::size_t SineSource::read(std::int16_t* out, std::size_t frames) {
    if (produced_ >= total_frames_) return 0;
    const std::uint32_t want = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(frames), total_frames_ - produced_);
    constexpr double TWO_PI = 6.283185307179586;
    const double f_l = freq_hz_;
    const double f_r = freq_hz_ + 10.0;
    const double sr  = static_cast<double>(samplerate_);
    for (std::uint32_t k = 0; k < want; ++k) {
        const double t = static_cast<double>(produced_ + k) / sr;
        const int l = static_cast<int>(std::round(16000.0 * std::sin(TWO_PI * f_l * t)));
        const int r = static_cast<int>(std::round(16000.0 * std::sin(TWO_PI * f_r * t)));
        out[2*k    ] = static_cast<std::int16_t>(std::clamp(l, -32768, 32767));
        out[2*k + 1] = static_cast<std::int16_t>(std::clamp(r, -32768, 32767));
    }
    produced_ += want;
    return want;
}

} // namespace rmps::wav
