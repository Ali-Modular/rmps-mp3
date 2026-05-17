#include "rmps/rmps.hpp"
#include "rmps/version.hpp"
#include "tui.hpp"
#include "wav_io.hpp"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

struct CliOpts {
    std::string input;            // empty => synth sine
    std::string output = "out.mp3";
    int    bitrate     = 128;
    double duration_s  = 2.0;
    double freq_hz     = 440.0;
    std::optional<double> shaping;
    std::optional<int>    abr_kbps;
    std::optional<int>    abr_min;
    std::optional<int>    abr_max;
    std::optional<double> target_sdr_db;
    std::optional<std::string> pid;
};

void usage(std::FILE* out) {
    std::fprintf(out,
        "rmps %s - rate-managed perceptual MP3 encoder\n"
        "\n"
        "USAGE\n"
        "  rmps [options]\n"
        "\n"
        "INPUT / OUTPUT\n"
        "  -i, --input  PATH       16-bit PCM WAV. Omit for a 2 s 440 Hz sine.\n"
        "  -o, --output PATH       MP3 output. Use '-' to stream to stdout.\n"
        "  -b, --bitrate KBPS      CBR target (ignored when --abr is set). [128]\n"
        "  -d, --duration SECS     synth duration (no effect with --input). [2.0]\n"
        "      --freq HZ           synth left-channel frequency. [440]\n"
        "\n"
        "QUALITY KNOBS\n"
        "      --shaping ALPHA     noise-shaping coefficient in [0..0.95]. [0.5]\n"
        "\n"
        "ABR / PID\n"
        "      --abr KBPS          enable ABR with target average bitrate.\n"
        "      --abr-min KBPS      lower clamp (default target/2).\n"
        "      --abr-max KBPS      upper clamp (default target*8/5).\n"
        "      --target-sdr DB     defend a perceptual SDR floor when ABR is on.\n"
        "      --pid Kp,Ki,Kd      override PID gains. [0.5,0.12,0.04]\n"
        "\n"
        "MISC\n"
        "  -h, --help              show this help.\n"
        "      --version           print version and exit.\n",
        rmps::VERSION_STRING);
}

bool parse_int(const std::string& s, int& out) {
    try { out = std::stoi(s); return true; } catch (...) { return false; }
}
bool parse_double(const std::string& s, double& out) {
    try { out = std::stod(s); return true; } catch (...) { return false; }
}

int parse_args(int argc, char** argv, CliOpts& opts) {
    auto need = [&](int i)->const char*{
        if (i + 1 >= argc) {
            std::fprintf(stderr, "rmps: missing value for %s\n", argv[i]);
            return nullptr;
        }
        return argv[i + 1];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage(stdout); std::exit(0); }
        if (a == "--version") {
            std::printf("rmps %s\n", rmps::VERSION_STRING); std::exit(0);
        }
        if (a == "-i" || a == "--input")    { auto v = need(i); if (!v) return 2; opts.input    = v; ++i; }
        else if (a == "-o" || a == "--output"){ auto v = need(i); if (!v) return 2; opts.output  = v; ++i; }
        else if (a == "-b" || a == "--bitrate") { auto v = need(i); if (!v) return 2; if (!parse_int(v, opts.bitrate)) return 2; ++i; }
        else if (a == "-d" || a == "--duration"){ auto v = need(i); if (!v) return 2; if (!parse_double(v, opts.duration_s)) return 2; ++i; }
        else if (a == "--freq")             { auto v = need(i); if (!v) return 2; if (!parse_double(v, opts.freq_hz)) return 2; ++i; }
        else if (a == "--shaping")          { auto v = need(i); if (!v) return 2; double d; if (!parse_double(v, d)) return 2; opts.shaping = d; ++i; }
        else if (a == "--abr")              { auto v = need(i); if (!v) return 2; int d; if (!parse_int(v, d)) return 2; opts.abr_kbps = d; ++i; }
        else if (a == "--abr-min")          { auto v = need(i); if (!v) return 2; int d; if (!parse_int(v, d)) return 2; opts.abr_min = d; ++i; }
        else if (a == "--abr-max")          { auto v = need(i); if (!v) return 2; int d; if (!parse_int(v, d)) return 2; opts.abr_max = d; ++i; }
        else if (a == "--target-sdr")       { auto v = need(i); if (!v) return 2; double d; if (!parse_double(v, d)) return 2; opts.target_sdr_db = d; ++i; }
        else if (a == "--pid")              { auto v = need(i); if (!v) return 2; opts.pid = std::string(v); ++i; }
        else {
            std::fprintf(stderr, "rmps: unknown option %s (try --help)\n", a.c_str());
            return 2;
        }
    }
    return 0;
}

// Opaque output stream wrapper: file or stdout binary.
struct OutSink {
    std::ofstream file;
    bool          is_stdout = false;

    void write(const std::uint8_t* data, std::size_t n) {
        if (is_stdout) std::cout.write(reinterpret_cast<const char*>(data),
                                       static_cast<std::streamsize>(n));
        else           file.write(reinterpret_cast<const char*>(data),
                                  static_cast<std::streamsize>(n));
    }
    void flush() { if (is_stdout) std::cout.flush(); else file.flush(); }
};

std::unique_ptr<OutSink> open_output(const std::string& path) {
    auto s = std::make_unique<OutSink>();
    if (path == "-") {
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        s->is_stdout = true;
    } else {
        s->file.open(path, std::ios::binary);
        if (!s->file) throw std::runtime_error("cannot open output: " + path);
    }
    return s;
}

} // namespace

int main(int argc, char** argv) {
    CliOpts opts;
    if (int rc = parse_args(argc, argv, opts); rc != 0) return rc;

    std::ostream& ui_stream = (opts.output == "-") ? std::cerr : std::cout;

    // ----- pick PCM source -----
    std::unique_ptr<rmps::wav::Reader>     wav_src;
    std::unique_ptr<rmps::wav::SineSource> sine_src;
    rmps::wav::WavInfo info {};
    std::string source_label;

    try {
        if (!opts.input.empty()) {
            wav_src = rmps::wav::Reader::open(opts.input);
            info = wav_src->info();
            source_label = std::filesystem::path(opts.input).filename().string();
        } else {
            sine_src = std::make_unique<rmps::wav::SineSource>(
                opts.freq_hz, 44100, opts.duration_s);
            info = sine_src->info();
            char buf[64];
            std::snprintf(buf, sizeof(buf), "synth %.0f Hz / %.0f Hz",
                          opts.freq_hz, opts.freq_hz + 10.0);
            source_label = buf;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "rmps: %s\n", e.what());
        return 1;
    }

    // ----- open output -----
    std::unique_ptr<OutSink> sink;
    try { sink = open_output(opts.output); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "rmps: %s\n", e.what()); return 1;
    }

    // ----- TUI banner (Phase 1: encoder not implemented yet) -----
    rmps::tui::Live::Options to;
    to.source_label = source_label;
    to.samplerate   = info.samplerate;
    to.channels     = info.channels;
    to.duration_s   = info.duration_s;
    to.bitrate_kbps = opts.bitrate;
    to.mode         = opts.abr_kbps
        ? ("ABR target " + std::to_string(*opts.abr_kbps) + " kbps"
            + (opts.target_sdr_db
                ? (", SDR floor " + std::to_string(*opts.target_sdr_db) + " dB") : ""))
        : ("CBR " + std::to_string(opts.bitrate) + " kbps");
    to.accel        = "C++20, -O3 -ffast-math";
    rmps::tui::Live live(ui_stream, std::move(to));

    // ----- create encoder -----
    rmps::Config ccfg;
    ccfg.channels   = info.channels;
    ccfg.samplerate = info.samplerate;
    ccfg.bitrate    = opts.bitrate;
    if (opts.abr_kbps)      ccfg.abr_target_kbps = *opts.abr_kbps;
    if (opts.abr_min)       ccfg.abr_min_kbps    = *opts.abr_min;
    if (opts.abr_max)       ccfg.abr_max_kbps    = *opts.abr_max;
    if (opts.target_sdr_db) ccfg.target_sdr_db   = *opts.target_sdr_db;
    if (opts.shaping)       ccfg.noise_shaping_alpha =
        std::clamp(*opts.shaping, 0.0, 0.95);
    auto encoder = rmps::Encoder::create(ccfg);
    if (!encoder) {
        std::fprintf(stderr, "rmps: unsupported samplerate/bitrate combo\n");
        return 1;
    }
    const std::size_t spp = encoder->samples_per_pass();
    const std::size_t chunk_interleaved = spp * static_cast<std::size_t>(info.channels);

    std::vector<std::int16_t> pcm_buf;
    pcm_buf.reserve(chunk_interleaved * 4);
    std::vector<std::int16_t> read_buf(spp * static_cast<std::size_t>(info.channels));

    const std::size_t total_frames =
        (info.total_frames + spp - 1u) / std::max<std::size_t>(spp, 1u);
    std::size_t frames_done = 0;
    std::size_t bytes_out   = 0;

    // Rolling SDR aggregates so the final summary isn't dragged to 0 by a
    // trailing zero-padded frame.
    double sdr_raw_sum = 0.0, sdr_perc_sum = 0.0;
    std::size_t sdr_n = 0;

    auto t0 = std::chrono::steady_clock::now();
    while (true) {
        std::size_t got_frames = wav_src
            ? wav_src->read(read_buf.data(), spp)
            : sine_src->read(read_buf.data(), spp);
        if (got_frames == 0) break;
        // Pad short final reads with zeros.
        const std::size_t got_samples = got_frames * static_cast<std::size_t>(info.channels);
        pcm_buf.insert(pcm_buf.end(), read_buf.begin(), read_buf.begin() + got_samples);
        if (got_frames < spp) {
            pcm_buf.resize(pcm_buf.size() + (spp - got_frames) * info.channels, 0);
        }

        while (pcm_buf.size() >= chunk_interleaved) {
            rmps::PcmChunk pc { pcm_buf.data(), chunk_interleaved };
            auto encoded = encoder->encode(pc);
            sink->write(encoded.data(), encoded.size());
            sink->flush();
            bytes_out += encoded.size();
            pcm_buf.erase(pcm_buf.begin(),
                          pcm_buf.begin() + static_cast<std::ptrdiff_t>(chunk_interleaved));
            ++frames_done;

            rmps::FrameStats fs = encoder->last_frame_stats();
            // Replace approximated total_bits with the actually emitted size
            // (includes header + side info + main data + stuffing).
            fs.total_bits = static_cast<int>(encoded.size() * 8u);
            if (!fs.granules.empty()) {
                sdr_raw_sum  += fs.mean_raw_sdr_db;
                sdr_perc_sum += fs.mean_perc_sdr_db;
                ++sdr_n;
            }
            live.on_frame(fs, encoder->abr_state(), bytes_out,
                          frames_done, total_frames);
        }
    }
    auto flushed = encoder->flush();
    if (!flushed.empty()) {
        sink->write(flushed.data(), flushed.size());
        sink->flush();
        bytes_out += flushed.size();
    }

    const double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double avg_raw  = sdr_n ? sdr_raw_sum  / static_cast<double>(sdr_n) : 0.0;
    const double avg_perc = sdr_n ? sdr_perc_sum / static_cast<double>(sdr_n) : 0.0;
    live.on_finish(elapsed_s, bytes_out, frames_done,
                   avg_raw, avg_perc, encoder->abr_state());

    return 0;
}
