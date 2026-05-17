#pragma once
#include "rmps/stats.hpp"

#include <chrono>
#include <cstdint>
#include <iosfwd>
#include <string>

namespace rmps::tui {

/// Minimal ANSI TUI: rich-style boxes, progress bar, live stats. Writes to
/// the chosen stream (default stderr so MP3 bytes can flow on stdout).
class Live {
public:
    struct Options {
        std::string source_label;
        int    samplerate = 0;
        int    channels   = 0;
        double duration_s = 0.0;
        int    bitrate_kbps = 0;
        std::string mode;             ///< "CBR 128 kbps" / "ABR target 128"
        std::string accel;            ///< "C++17 / -O3"
        bool   ansi_enabled = true;
    };

    explicit Live(std::ostream& out, Options opts);
    ~Live();

    /// Call once per encoded frame.
    void on_frame(const FrameStats& fs,
                  const AbrState*   abr,
                  std::size_t       total_bytes,
                  std::size_t       frames_done,
                  std::size_t       frames_total);

    /// Final summary block, called once after flush.
    void on_finish(double elapsed_s,
                   std::size_t total_bytes,
                   std::size_t frames_done,
                   double final_raw_sdr_db,
                   double final_perc_sdr_db,
                   const AbrState* abr);

private:
    void render_header();
    void clear_lines(int n);

    std::ostream& out_;
    Options       opts_;
    int           drawn_lines_ = 0;
    std::chrono::steady_clock::time_point t_start_;
};

} // namespace rmps::tui
