#include "rmps/rmps.hpp"
#include "rmps/version.hpp"
#include "audio_in.hpp"
#include "encode_job.hpp"
#include "tui.hpp"
#include "utf8.hpp"

#ifdef RMPS_HAVE_GUI
#include "gui.hpp"
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

struct CliOpts {
    std::string input;             // empty => synth sine
    std::string output = "out.mp3";
    int    bitrate      = 128;
    bool   cbr_explicit = false;   // user passed -b/--bitrate
    double duration_s   = 2.0;
    double freq_hz      = 440.0;
    std::optional<int>    abr_kbps;
    std::optional<int>    abr_min;
    std::optional<int>    abr_max;
    std::optional<double> quality_sdr_db;   // -q
    std::optional<int>    effort;           // -e: search effort 0..10
    bool   hf_extension = false;           // --hf-ext
};

void usage(std::FILE* out) {
    std::fprintf(out,
        "rmps %s - rate-managed perceptual MP3 encoder\n"
        "\n"
        "USAGE\n"
        "  rmps [options]          encode from the command line\n"
        "  rmps                    (no arguments) open the GUI\n"
        "\n"
        "INPUT / OUTPUT\n"
        "  -i, --input  PATH       audio file: %s.\n"
        "                          Omit for a 2 s 440 Hz sine.\n"
        "  -o, --output PATH       MP3 output. Use '-' to stream to stdout.\n"
        "  -d, --duration SECS     synth duration (no effect with --input). [2.0]\n"
        "      --freq HZ           synth left-channel frequency. [440]\n"
        "\n"
        "QUALITY\n"
        "  -q, --quality N         1..100. Sets the perceptual score the encoder\n"
        "                          holds - 10 dB at 1, 32 dB at 100 - and moves\n"
        "                          the bitrate wherever that needs, over the whole\n"
        "                          legal range.\n"
        "  -e, --effort N          0..10 search effort. [2] Higher widens the\n"
        "                          quantizer search (scalefactor range, gain\n"
        "                          window, rate probes, polish passes) at the cost\n"
        "                          of encode time. The scalefactor range is what\n"
        "                          lets quiet bands next to loud ones survive:\n"
        "                          1 at effort 0-2, 2 at 3, 3 at 4, 4 at 5-6,\n"
        "                          5 at 7+.\n"
        "\n"
        "RATE CONTROL  (ignored when --quality is given)\n"
        "      --abr KBPS          ABR target average bitrate.\n"
        "  -b, --bitrate KBPS      CBR at KBPS (opt out of ABR). [128]\n"
        "      --abr-min KBPS      lower clamp (default target-12).\n"
        "      --abr-max KBPS      upper clamp (default target+48).\n"
        "\n"
        "MISC\n"
        "  -h, --help              show this help.\n"
        "      --version           print version and exit.\n"
        "      --hf-ext            enable online high-frequency bandwidth extension.\n",
        rmps::VERSION_STRING, rmps::app::supported_formats());
}

bool parse_int(const std::string& s, int& out) {
    try { out = std::stoi(s); return true; } catch (...) { return false; }
}
bool parse_double(const std::string& s, double& out) {
    try { out = std::stod(s); return true; } catch (...) { return false; }
}

int parse_args(const std::vector<std::string>& args, CliOpts& opts) {
    const std::size_t argc = args.size();
    auto need = [&](std::size_t i) -> const std::string* {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "rmps: missing value for %s\n", args[i].c_str());
            return nullptr;
        }
        return &args[i + 1];
    };
    for (std::size_t i = 1; i < argc; ++i) {
        const std::string& a = args[i];
        if (a == "-h" || a == "--help") { usage(stdout); std::exit(0); }
        if (a == "--version") { std::printf("rmps %s\n", rmps::VERSION_STRING); std::exit(0); }
        if (a == "-i" || a == "--input")        { auto v = need(i); if (!v) return 2; opts.input = *v; ++i; }
        else if (a == "-o" || a == "--output")  { auto v = need(i); if (!v) return 2; opts.output = *v; ++i; }
        else if (a == "-b" || a == "--bitrate") { auto v = need(i); if (!v) return 2; if (!parse_int(*v, opts.bitrate)) return 2; opts.cbr_explicit = true; ++i; }
        else if (a == "-d" || a == "--duration"){ auto v = need(i); if (!v) return 2; if (!parse_double(*v, opts.duration_s)) return 2; ++i; }
        else if (a == "--freq")                 { auto v = need(i); if (!v) return 2; if (!parse_double(*v, opts.freq_hz)) return 2; ++i; }
        else if (a == "-q" || a == "--quality") {
            auto v = need(i); if (!v) return 2;
            double d; if (!parse_double(*v, d)) return 2;
            if (d < 1.0 || d > 100.0) {
                std::fprintf(stderr, "rmps: --quality must be 1..100\n");
                return 2;
            }
            // 1..100 maps linearly onto the quality score the encoder holds.
            //
            // The endpoints are the measured reach of the encoder on real
            // program material, not round numbers. Pinned-rate Fullness /
            // Bleedless scores over four 30 s excerpts (`ab_listen/*`):
            //
            //     kbps    32     64    128    320
            //     range  7.0   13.1   19.2   32.9
            //            11.2  19.2   27.6   43.7
            //
            // The first calibration used the full 7..44 dB reach. 44 dB was
            // only the single easiest excerpt (shishamo); the other three
            // topped out at 32.9-35.4 dB at 320 kbps, so on ordinary material
            // the top third of that scale asked for a score no rate delivers.
            // 10..32 dB keeps both ends codeable: measured on the four
            // excerpts, -q 1 gives 34-48 kbps, -q 50 76-149, -q 100 167-287,
            // monotone throughout and never pinned at 320.
            opts.quality_sdr_db = rmps::app::quality_to_score_db(d);
            ++i;
        }
        else if (a == "-e" || a == "--effort") {
            auto v = need(i); if (!v) return 2;
            int d; if (!parse_int(*v, d)) return 2;
            if (d < 0 || d > 10) {
                std::fprintf(stderr, "rmps: --effort must be 0..10\n");
                return 2;
            }
            opts.effort = d;
            ++i;
        }
        else if (a == "--abr")     { auto v = need(i); if (!v) return 2; int d; if (!parse_int(*v, d)) return 2; opts.abr_kbps = d; ++i; }
        else if (a == "--abr-min") { auto v = need(i); if (!v) return 2; int d; if (!parse_int(*v, d)) return 2; opts.abr_min = d; ++i; }
        else if (a == "--abr-max") { auto v = need(i); if (!v) return 2; int d; if (!parse_int(*v, d)) return 2; opts.abr_max = d; ++i; }
        else if (a == "--hf-ext")  { opts.hf_extension = true; }
        else {
            std::fprintf(stderr, "rmps: unknown option %s (try --help)\n", a.c_str());
            return 2;
        }
    }
    return 0;
}

int run_cli(const std::vector<std::string>& args) {
    CliOpts opts;
    if (int rc = parse_args(args, opts); rc != 0) return rc;

    // Default to ABR unless the user explicitly opted into CBR via `-b`.
    if (!opts.cbr_explicit && !opts.abr_kbps) opts.abr_kbps = opts.bitrate;

    rmps::app::EncodeJob job;
    job.input = opts.input;
    job.output = opts.output;
    job.bitrate = opts.bitrate;
    job.abr_kbps = opts.abr_kbps;
    job.abr_min = opts.abr_min;
    job.abr_max = opts.abr_max;
    job.quality_sdr_db = opts.quality_sdr_db;
    job.effort = opts.effort;
    job.hf_extension = opts.hf_extension;
    job.synth_freq_hz = opts.freq_hz;
    job.synth_duration_s = opts.duration_s;

    std::ostream& ui = (opts.output == "-") ? std::cerr : std::cout;
    std::unique_ptr<rmps::tui::Live> live;

    rmps::app::EncodeCallbacks cb;
    cb.on_start = [&](const rmps::app::EncodeProgress& p) {
        if (p.coded_samplerate != p.input_samplerate)
            std::fprintf(stderr, "rmps: coding at %d Hz (input %d Hz)\n",
                         p.coded_samplerate, p.input_samplerate);
        rmps::tui::Live::Options to;
        to.source_label = p.source_label;
        to.samplerate = p.input_samplerate;
        to.channels = p.source_channels;
        to.duration_s = p.duration_s;
        to.bitrate_kbps = opts.bitrate;
        to.mode = opts.quality_sdr_db
            ? ("VBR quality, target " + std::to_string(static_cast<int>(*opts.quality_sdr_db)) + " dB")
            : opts.abr_kbps ? ("ABR target " + std::to_string(*opts.abr_kbps) + " kbps")
                            : ("CBR " + std::to_string(opts.bitrate) + " kbps");
        if (opts.effort) to.mode += ", effort " + std::to_string(*opts.effort);
        to.accel = p.source_format;
        live = std::make_unique<rmps::tui::Live>(ui, std::move(to));
    };
    cb.on_frame = [&](const rmps::app::EncodeProgress& p, const rmps::FrameStats& fs,
                      const rmps::AbrState* abr) {
        if (live) live->on_frame(fs, abr, p.bytes_out, p.mp3_frames_done, p.mp3_frames_total);
    };

    const auto res = rmps::app::run_encode(job, cb);
    if (!res.ok) {
        std::fprintf(stderr, "rmps: %s\n", res.error.c_str());
        return 1;
    }
    if (live)
        live->on_finish(res.elapsed_s, res.bytes_out, res.mp3_frames,
                        res.avg_raw_sdr_db, res.avg_perc_sdr_db,
                        res.abr ? &*res.abr : nullptr);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const auto args = rmps::app::utf8_args(argc, argv);
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
#ifdef RMPS_HAVE_GUI
    if (args.size() <= 1) {
#ifdef _WIN32
        // Started without arguments. When the process owns its console — it was
        // double-clicked, so Windows made one just for it — drop that console
        // so only the GUI window remains. Started from a terminal, the console
        // belongs to the shell and is left alone.
        DWORD pids[2];
        if (GetConsoleProcessList(pids, 2) == 1) FreeConsole();
#endif
        return rmps::app::run_gui();
    }
#endif
    return run_cli(args);
}
