#include "encode_job.hpp"
#include "audio_in.hpp"
#include "utf8.hpp"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

namespace rmps::app {

int coded_rate_for_input(int sr) {
    if (sr == 32000 || sr == 44100 || sr == 48000) return sr;
    // Everything else is taken to an MPEG-1 rate. Rates below 32 kHz would
    // otherwise select MPEG-2 / 2.5, whose LSF scalefactor coding this encoder
    // does not implement — such a stream decodes as garbage. Upsampling a
    // low-rate source costs bits but produces a valid file.
    if (sr % 11025 == 0) return 44100;   // 11.025 / 22.05 / 88.2 / 176.4 kHz
    return 48000;                        // 8 / 12 / 16 / 24 / 96 / 192 kHz, other
}

namespace {

struct Sink {
    std::ofstream file;
    bool is_stdout = false;
    void write(const std::uint8_t* d, std::size_t n) {
        if (is_stdout) std::cout.write(reinterpret_cast<const char*>(d), static_cast<std::streamsize>(n));
        else file.write(reinterpret_cast<const char*>(d), static_cast<std::streamsize>(n));
    }
    bool good() const { return is_stdout ? static_cast<bool>(std::cout) : static_cast<bool>(file); }
};

} // namespace

EncodeResult run_encode(const EncodeJob& job, const EncodeCallbacks& cb) {
    EncodeResult res;
    // Set once this run has created the output file, so a failure after that
    // point does not leave a truncated or empty MP3 behind.
    bool created_output = false;
    try {
        std::unique_ptr<AudioSource> src;
        std::string label;
        if (!job.input.empty()) {
            src = open_audio(job.input);
            label = from_path(to_path(job.input).filename());
        } else {
            src = make_sine(job.synth_freq_hz, 44100, job.synth_duration_s);
            char buf[64];
            std::snprintf(buf, sizeof buf, "synth %.0f Hz / %.0f Hz",
                          job.synth_freq_hz, job.synth_freq_hz + 10.0);
            label = buf;
        }
        const AudioInfo info = src->info();

        Sink sink;
        if (job.output == "-") {
#ifdef _WIN32
            _setmode(_fileno(stdout), _O_BINARY);
#endif
            sink.is_stdout = true;
        } else {
            // Opening through a wide path is what makes non-ASCII output
            // folders work.
            sink.file.open(to_path(job.output), std::ios::binary);
            if (!sink.file)
                throw std::runtime_error("cannot open output: " + job.output);
            created_output = true;
        }

        Config cfg;
        cfg.channels = info.channels;
        cfg.samplerate = info.samplerate;
        cfg.bitrate = job.bitrate;
        if (job.abr_kbps) cfg.abr_target_kbps = *job.abr_kbps;
        if (job.abr_min) cfg.abr_min_kbps = *job.abr_min;
        if (job.abr_max) cfg.abr_max_kbps = *job.abr_max;
        if (job.quality_sdr_db) cfg.quality_sdr_db = *job.quality_sdr_db;
        if (job.effort) cfg.effort = *job.effort;
        cfg.hf_extension = job.hf_extension;
        const int coded = coded_rate_for_input(info.samplerate);
        if (coded != info.samplerate) cfg.coded_samplerate = coded;

        auto encoder = Encoder::create(cfg);
        if (!encoder)
            throw std::runtime_error("unsupported samplerate/bitrate combination");

        EncodeProgress prog;
        prog.input_frames_total = info.total_frames;
        prog.duration_s = info.duration_s;
        prog.input_samplerate = info.samplerate;
        prog.coded_samplerate = encoder->coded_samplerate();
        prog.source_channels = info.source_channels;
        prog.source_label = label;
        prog.source_format = info.format;
        const std::size_t spp = encoder->samples_per_pass();
        prog.mp3_frames_total = static_cast<std::size_t>(
            (static_cast<double>(info.total_frames) * prog.coded_samplerate /
                 std::max(info.samplerate, 1) + static_cast<double>(spp) - 1.0) /
            static_cast<double>(spp));
        if (cb.on_start) cb.on_start(prog);

        const std::size_t ch = static_cast<std::size_t>(info.channels);
        const std::size_t chunk = spp * ch;
        std::vector<std::int16_t> pcm, rd(chunk);
        pcm.reserve(chunk * 4);
        double raw_sum = 0.0, perc_sum = 0.0;
        std::size_t sdr_n = 0;
        const auto t0 = std::chrono::steady_clock::now();

        auto emit = [&](std::span<const std::uint8_t> bytes) {
            if (bytes.empty()) return;
            sink.write(bytes.data(), bytes.size());
            res.bytes_out += bytes.size();
        };

        for (;;) {
            if (cb.cancel && cb.cancel->load()) { res.cancelled = true; break; }
            const std::size_t got = src->read(rd.data(), spp);
            if (got == 0) break;
            prog.input_frames_done += got;
            pcm.insert(pcm.end(), rd.begin(), rd.begin() + static_cast<std::ptrdiff_t>(got * ch));
            if (got < spp) pcm.resize(pcm.size() + (spp - got) * ch, 0);
            while (pcm.size() >= chunk) {
                auto enc = encoder->encode(PcmChunk{pcm.data(), chunk});
                emit(enc);
                pcm.erase(pcm.begin(), pcm.begin() + static_cast<std::ptrdiff_t>(chunk));
                ++prog.mp3_frames_done;
                FrameStats fs = encoder->last_frame_stats();
                fs.total_bits = static_cast<int>(enc.size() * 8u);
                if (!fs.granules.empty()) {
                    raw_sum += fs.mean_raw_sdr_db;
                    perc_sum += fs.mean_perc_sdr_db;
                    ++sdr_n;
                }
                prog.bytes_out = res.bytes_out;
                prog.elapsed_s = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - t0).count();
                if (cb.on_frame) cb.on_frame(prog, fs, encoder->abr_state());
            }
        }
        if (prog.input_frames_done == 0 && !res.cancelled)
            // A file that opens but yields no audio (truncated, or a container
            // with no decodable stream) would otherwise "succeed" with an
            // empty MP3.
            throw std::runtime_error("no audio could be decoded from: " + job.input);
        emit(encoder->flush());
        auto tag = encoder->vbr_tag();
        if (!tag.empty() && !sink.is_stdout) {
            sink.file.seekp(0);
            sink.file.write(reinterpret_cast<const char*>(tag.data()),
                            static_cast<std::streamsize>(tag.size()));
        }
        if (sink.is_stdout) std::cout.flush(); else sink.file.flush();
        if (!sink.good()) throw std::runtime_error("write failed: " + job.output);

        res.mp3_frames = prog.mp3_frames_done;
        res.elapsed_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
        res.duration_s = info.duration_s;
        res.avg_raw_sdr_db = sdr_n ? raw_sum / static_cast<double>(sdr_n) : 0.0;
        res.avg_perc_sdr_db = sdr_n ? perc_sum / static_cast<double>(sdr_n) : 0.0;
        if (const AbrState* a = encoder->abr_state()) res.abr = *a;
        res.ok = !res.cancelled;
        if (res.cancelled && !sink.is_stdout) {
            sink.file.close();
            std::error_code ec;
            std::filesystem::remove(to_path(job.output), ec);
        }
    } catch (const std::exception& e) {
        res.ok = false;
        res.error = e.what();
        if (created_output) {
            std::error_code ec;
            std::filesystem::remove(to_path(job.output), ec);
        }
    }
    return res;
}

} // namespace rmps::app
