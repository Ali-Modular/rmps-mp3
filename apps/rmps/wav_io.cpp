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
            if (chunk_sz > 16) r->file_.seekg(chunk_sz - 16, std::ios::cur);
            if (fmt_tag != 1)
                throw std::runtime_error("only PCM (fmt 1) WAV supported");
            if (bps != 16)
                throw std::runtime_error("only 16-bit WAV supported");
            if (r->src_channels_ < 1 || r->src_channels_ > 2)
                throw std::runtime_error("only mono/stereo WAV supported");
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
    const std::uint32_t frame_bytes = static_cast<std::uint32_t>(src_ch) * 2u;
    r->info_.total_frames = r->data_remaining_bytes_ / frame_bytes;
    r->info_.channels     = 2;       // we always emit stereo
    r->info_.duration_s   = static_cast<double>(r->info_.total_frames)
                          / static_cast<double>(r->info_.samplerate);
    return r;
}

std::size_t Reader::read(std::int16_t* out, std::size_t frames) {
    if (data_remaining_bytes_ == 0) return 0;
    const std::uint32_t frame_bytes = static_cast<std::uint32_t>(src_channels_) * 2u;
    const std::uint32_t max_frames  = data_remaining_bytes_ / frame_bytes;
    if (frames > max_frames) frames = max_frames;
    if (frames == 0) return 0;

    if (src_channels_ == 2) {
        file_.read(reinterpret_cast<char*>(out),
                   static_cast<std::streamsize>(frames * frame_bytes));
        data_remaining_bytes_ -= static_cast<std::uint32_t>(frames * frame_bytes);
        return frames;
    }
    // mono -> stereo up-mix via duplicate
    if (scratch_.size() < frames) scratch_.resize(frames);
    file_.read(reinterpret_cast<char*>(scratch_.data()),
               static_cast<std::streamsize>(frames * 2u));
    data_remaining_bytes_ -= static_cast<std::uint32_t>(frames * 2u);
    for (std::size_t i = 0; i < frames; ++i) {
        out[2*i    ] = scratch_[i];
        out[2*i + 1] = scratch_[i];
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
