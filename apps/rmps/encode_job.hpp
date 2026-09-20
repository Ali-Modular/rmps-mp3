#pragma once
#include "rmps/rmps.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace rmps::app {

/// One encode, described independently of how it was requested (CLI or GUI).
struct EncodeJob {
    std::string input;            ///< UTF-8 path; empty = synth sine
    std::string output;           ///< UTF-8 path; "-" = stdout
    int bitrate = 128;            ///< CBR rate, or the ABR target when abr is set
    std::optional<int> abr_kbps;
    std::optional<int> abr_min;
    std::optional<int> abr_max;
    std::optional<double> quality_sdr_db;
    std::optional<int> effort;
    bool hf_extension = false;
    double synth_freq_hz = 440.0;
    double synth_duration_s = 2.0;
};

struct EncodeProgress {
    std::uint64_t input_frames_done = 0;
    std::uint64_t input_frames_total = 0;  ///< 0 when unknown
    std::size_t mp3_frames_done = 0;
    std::size_t mp3_frames_total = 0;      ///< estimate
    std::size_t bytes_out = 0;
    double elapsed_s = 0.0;
    double duration_s = 0.0;               ///< of the source
    int input_samplerate = 0;
    int coded_samplerate = 0;
    int source_channels = 0;
    std::string source_label;
    std::string source_format;
};

struct EncodeResult {
    bool ok = false;
    bool cancelled = false;
    std::string error;
    std::size_t bytes_out = 0;
    std::size_t mp3_frames = 0;
    double elapsed_s = 0.0;
    double duration_s = 0.0;
    double avg_raw_sdr_db = 0.0;
    double avg_perc_sdr_db = 0.0;
    std::optional<AbrState> abr;
};

struct EncodeCallbacks {
    /// Called once the source is open and the encoder created, before any
    /// frame is coded. Lets the caller draw a header.
    std::function<void(const EncodeProgress&)> on_start;
    /// Called per emitted MP3 frame.
    std::function<void(const EncodeProgress&, const FrameStats&, const AbrState*)> on_frame;
    /// Set to true from any thread to stop at the next frame boundary.
    const std::atomic<bool>* cancel = nullptr;
};

/// Run the job end to end: open, encode, flush, finalise the Xing/Info tag.
/// Never throws; failures come back in `EncodeResult::error`.
EncodeResult run_encode(const EncodeJob& job, const EncodeCallbacks& cb);

/// The MPEG-1 rate a source at `samplerate` is coded at: itself for 32, 44.1
/// and 48 kHz, otherwise the MPEG-1 member of its rate family.
int coded_rate_for_input(int samplerate);

/// `-q 1..100` to the quality score (dB) the encoder holds. Shared by the CLI
/// and the GUI so both mean the same thing by the same number. 10 dB at 1,
/// 32 dB at 100; see the calibration note in main.cpp.
inline double quality_to_score_db(double q) {
    constexpr double LO = 10.0, HI = 32.0;
    return LO + (q - 1.0) * (HI - LO) / 99.0;
}

} // namespace rmps::app
