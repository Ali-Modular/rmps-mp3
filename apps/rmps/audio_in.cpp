#include "audio_in.hpp"
#include "utf8.hpp"
#include "wav_io.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef RMPS_HAVE_SNDFILE
#ifdef _WIN32
#define ENABLE_SNDFILE_WINDOWS_PROTOTYPES 1
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include <sndfile.h>
#endif

namespace rmps::app {

namespace {

#ifdef RMPS_HAVE_SNDFILE
class SndfileSource final : public AudioSource {
public:
    explicit SndfileSource(const std::string& utf8_path) {
        SF_INFO si{};
#ifdef _WIN32
        sf_ = sf_wchar_open(widen(utf8_path).c_str(), SFM_READ, &si);
#else
        sf_ = sf_open(utf8_path.c_str(), SFM_READ, &si);
#endif
        if (!sf_)
            throw std::runtime_error("cannot open input: " + utf8_path + " (" +
                                     sf_strerror(nullptr) + ")");
        if (si.channels < 1 || si.samplerate <= 0) {
            sf_close(sf_);
            throw std::runtime_error("unsupported stream: " + utf8_path);
        }
        // Float and Opus/Vorbis decode to float internally; let libsndfile
        // scale them to the int16 range without wrapping on overs.
        sf_command(sf_, SFC_SET_CLIPPING, nullptr, SF_TRUE);
        src_ch_ = si.channels;
        info_.source_channels = si.channels;
        info_.samplerate = si.samplerate;
        info_.total_frames = si.frames > 0 ? static_cast<std::uint64_t>(si.frames) : 0;
        info_.duration_s = static_cast<double>(info_.total_frames) / si.samplerate;
        SF_FORMAT_INFO fi{};
        fi.format = si.format & SF_FORMAT_TYPEMASK;
        std::string major = (sf_command(sf_, SFC_GET_FORMAT_INFO, &fi, sizeof fi) == 0 && fi.name)
                                ? fi.name : "audio";
        SF_FORMAT_INFO sub{};
        sub.format = si.format & SF_FORMAT_SUBMASK;
        std::string minor = (sf_command(sf_, SFC_GET_FORMAT_INFO, &sub, sizeof sub) == 0 && sub.name)
                                ? sub.name : "";
        info_.format = minor.empty() ? major : major + " / " + minor;
    }
    ~SndfileSource() override { if (sf_) sf_close(sf_); }

    const AudioInfo& info() const noexcept override { return info_; }

    std::size_t read(std::int16_t* out, std::size_t frames) override {
        buf_.resize(frames * static_cast<std::size_t>(src_ch_));
        const sf_count_t got =
            sf_readf_short(sf_, buf_.data(), static_cast<sf_count_t>(frames));
        if (got <= 0) return 0;
        const std::size_t n = static_cast<std::size_t>(got);
        const std::size_t C = static_cast<std::size_t>(src_ch_);
        for (std::size_t f = 0; f < n; ++f) {
            const short* s = buf_.data() + f * C;
            double l, r;
            if (C == 1) {
                l = r = s[0];
            } else if (C == 2) {
                l = s[0]; r = s[1];
            } else {
                // Fold-down: front pair at unity, centre (ch 3) at -3 dB into
                // both, everything else shared equally; normalised by the sum
                // of the weights so a full-scale input stays in range.
                double wl = 1.0, wr = 1.0;
                l = s[0]; r = s[1];
                const double c = 0.7071;
                l += c * s[2]; r += c * s[2]; wl += c; wr += c;
                for (std::size_t k = 3; k < C; ++k) {
                    l += 0.5 * s[k]; r += 0.5 * s[k]; wl += 0.5; wr += 0.5;
                }
                l /= wl; r /= wr;
            }
            out[2 * f] = static_cast<std::int16_t>(std::clamp(std::lround(l), -32768L, 32767L));
            out[2 * f + 1] = static_cast<std::int16_t>(std::clamp(std::lround(r), -32768L, 32767L));
        }
        return n;
    }

private:
    SNDFILE* sf_ = nullptr;
    int src_ch_ = 2;
    AudioInfo info_{};
    std::vector<short> buf_;
};
#endif

/// WAV-only fallback when built without libsndfile.
class WavSource final : public AudioSource {
public:
    explicit WavSource(const std::string& utf8_path)
        : r_(wav::Reader::open(from_path(to_path(utf8_path)))) {
        const auto& wi = r_->info();
        info_.channels = 2;
        info_.source_channels = wi.channels;
        info_.samplerate = wi.samplerate;
        info_.total_frames = wi.total_frames;
        info_.duration_s = wi.duration_s;
        info_.format = "WAV";
    }
    const AudioInfo& info() const noexcept override { return info_; }
    std::size_t read(std::int16_t* out, std::size_t frames) override {
        return r_->read(out, frames);
    }
private:
    std::unique_ptr<wav::Reader> r_;
    AudioInfo info_{};
};

class SineAdapter final : public AudioSource {
public:
    SineAdapter(double f, int sr, double d) : s_(f, sr, d) {
        const auto wi = s_.info();
        info_.channels = 2;
        info_.source_channels = 2;
        info_.samplerate = wi.samplerate;
        info_.total_frames = wi.total_frames;
        info_.duration_s = wi.duration_s;
        info_.format = "synth";
    }
    const AudioInfo& info() const noexcept override { return info_; }
    std::size_t read(std::int16_t* out, std::size_t frames) override {
        return s_.read(out, frames);
    }
private:
    wav::SineSource s_;
    AudioInfo info_{};
};

} // namespace

std::unique_ptr<AudioSource> open_audio(const std::string& utf8_path) {
#ifdef RMPS_HAVE_SNDFILE
    return std::make_unique<SndfileSource>(utf8_path);
#else
    return std::make_unique<WavSource>(utf8_path);
#endif
}

std::unique_ptr<AudioSource> make_sine(double freq_hz, int samplerate,
                                       double duration_s) {
    return std::make_unique<SineAdapter>(freq_hz, samplerate, duration_s);
}

const char* supported_formats() {
#ifdef RMPS_HAVE_SNDFILE
    return "WAV, FLAC, Ogg Vorbis, Ogg Opus, AIFF";
#else
    return "WAV";
#endif
}

} // namespace rmps::app
