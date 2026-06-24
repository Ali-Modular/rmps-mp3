// Define before ANY include so a transitive <windows.h> never defines the
// min()/max() macros that would clobber std::min / std::max used below.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "tui.hpp"

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <ostream>
#include <sstream>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

namespace rmps::tui {

namespace {

constexpr const char *RESET = "\x1b[0m";
constexpr const char *BOLD = "\x1b[1m";
constexpr const char *DIM = "\x1b[2m";
constexpr const char *CYAN = "\x1b[36m";
constexpr const char *MAGENTA = "\x1b[35m";
constexpr const char *GREEN = "\x1b[32m";
constexpr const char *YELLOW = "\x1b[33m";
constexpr const char *RED = "\x1b[31m";
constexpr const char *GRAY = "\x1b[90m";

void enable_vt_on_windows() {
#ifdef _WIN32
  HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
  DWORD mode = 0;
  if (h && GetConsoleMode(h, &mode))
    SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
  HANDLE h2 = GetStdHandle(STD_OUTPUT_HANDLE);
  if (h2 && GetConsoleMode(h2, &mode))
    SetConsoleMode(h2, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
}

std::string format_time(double s) {
  int total = static_cast<int>(s + 0.5);
  int m = total / 60;
  int sec = total % 60;
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%02d:%02d", m, sec);
  return std::string(buf);
}

} // namespace

Live::Live(std::ostream &out, Options opts)
    : out_(out), opts_(std::move(opts)) {
  if (opts_.ansi_enabled)
    enable_vt_on_windows();
  t_start_ = std::chrono::steady_clock::now();
  render_header();
}

Live::~Live() = default;

void Live::clear_lines(int n) {
  if (!opts_.ansi_enabled || n <= 0)
    return;
  for (int i = 0; i < n; ++i) {
    out_ << "\x1b[1A\x1b[2K";
  }
}

void Live::render_header() {
  const char *C = opts_.ansi_enabled ? CYAN : "";
  const char *B = opts_.ansi_enabled ? BOLD : "";
  const char *R = opts_.ansi_enabled ? RESET : "";
  out_ << B << C
       << "===================================================================="
          "==\n"
       << "  rmps  -  Rate-Managed Perceptual Encoder\n"
       << "===================================================================="
          "==\n"
       << R;
  out_ << "  source     : " << opts_.source_label << "\n";
  out_ << "  format     : 16-bit PCM @ " << opts_.samplerate << " Hz, "
       << opts_.channels << " ch\n";
  out_ << "  duration   : " << std::fixed << std::setprecision(2)
       << opts_.duration_s << " s\n";
  out_ << "  rate mode  : " << opts_.mode << "\n";
  out_ << "  accel      : " << opts_.accel << "\n";
  out_ << "--------------------------------------------------------------------"
          "--\n";
  out_.flush();
}

void Live::on_frame(const FrameStats &fs, const AbrState *abr,
                    std::size_t total_bytes, std::size_t frames_done,
                    std::size_t frames_total) {
  clear_lines(drawn_lines_);
  drawn_lines_ = 0;

  std::ostringstream s;
  s.setf(std::ios::fixed);
  s << std::setprecision(2);

  auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start_)
          .count();
  const double frac = frames_total ? static_cast<double>(frames_done) /
                                         static_cast<double>(frames_total)
                                   : 0.0;
  const double eta = (frac > 1e-6) ? elapsed * (1.0 - frac) / frac : 0.0;

  constexpr int BAR_W = 36;
  int filled = static_cast<int>(frac * BAR_W + 0.5);
  if (filled > BAR_W)
    filled = BAR_W;

  s << GREEN << "  [";
  for (int i = 0; i < filled; ++i)
    s << "#";
  for (int i = filled; i < BAR_W; ++i)
    s << "-";
  s << "] " << RESET << frames_done << "/" << frames_total << "  "
    << format_time(elapsed) << "  ETA " << format_time(eta) << "\n";

  s << "  " << CYAN << "frame SDR" << RESET << "   raw " << GREEN
    << std::setw(6) << fs.mean_raw_sdr_db << RESET << " dB   A-w " << CYAN
    << std::setw(6) << fs.mean_perc_sdr_db << RESET << " dB\n";
  s << "  " << CYAN << "bits" << RESET << "        " << fs.total_bits << " / "
    << fs.total_budget << "    " << CYAN << "bytes out " << RESET << total_bytes
    << "\n";

  if (abr) {
    double delta = abr->avg_kbps - abr->target_kbps;
    const char *col =
        std::abs(delta) < 4.0 ? GREEN : (std::abs(delta) < 12.0 ? YELLOW : RED);
    s << "  " << MAGENTA << "ABR" << RESET << "         target "
      << std::setprecision(0) << abr->target_kbps << " kbps  avg " << col
      << std::setprecision(2) << abr->avg_kbps << RESET << " ("
      << (delta >= 0 ? "+" : "") << delta << ")  next " << abr->decision_kbps
      << " kbps\n";
  }
  if (fs.lpf_side_hz > 0.0) {
    s << "  " << CYAN << "LPF M/S" << RESET << "     20.0 / "
      << std::setprecision(1) << (fs.lpf_side_hz / 1000.0) << " kHz\n";
  }
  if (fs.hf.atten_db >= 0.05) {
    s << "  " << CYAN << "HF shelf" << RESET << "    -" << std::setprecision(2)
      << fs.hf.atten_db << " dB @ " << std::setprecision(1)
      << (fs.hf.shelf_hz / 1000.0) << " kHz"
      << (fs.hf.transient ? (std::string(" ") + RED + "TRANSIENT" + RESET) : "")
      << "\n";
  }
  if (fs.hf_rdo_bits_saved) {
    s << "  " << CYAN << "HF RDO" << RESET << "      " << GREEN << "-"
      << fs.hf_rdo_bits_saved << " bits" << RESET << "\n";
  }
  if (fs.transient_granules > 0) {
    s << "  " << RED << "PRE-ECHO GATE" << RESET
      << "  " << fs.transient_granules << " granule"
      << (fs.transient_granules > 1 ? "s" : "")
      << " gated, cutoff " << std::setprecision(1)
      << (fs.transient_cutoff_hz / 1000.0) << " kHz\n";
  }
  if (fs.short_block_granules > 0 || fs.transition_block_granules > 0) {
    s << "  " << CYAN << "SHORT BLOCKS" << RESET
      << "   " << fs.short_block_granules << " short, "
      << fs.transition_block_granules << " transition\n";
  }

  const std::string buf = s.str();
  out_ << buf;
  out_.flush();
  drawn_lines_ = static_cast<int>(std::count(buf.begin(), buf.end(), '\n'));
}

void Live::on_finish(double elapsed_s, std::size_t total_bytes,
                     std::size_t frames_done, double final_raw_sdr_db,
                     double final_perc_sdr_db, const AbrState *abr) {
  clear_lines(drawn_lines_);
  drawn_lines_ = 0;
  const char *G = opts_.ansi_enabled ? GREEN : "";
  const char *R = opts_.ansi_enabled ? RESET : "";
  out_ << G
       << "--------------------------------------------------------------------"
          "--\n"
       << "  Encode summary\n"
       << "--------------------------------------------------------------------"
          "--\n"
       << R;
  out_ << "  frames        : " << frames_done << "\n";
  out_ << "  bytes written : " << total_bytes << "\n";
  out_ << std::fixed << std::setprecision(2);
  out_ << "  final raw SDR : " << final_raw_sdr_db << " dB\n";
  out_ << "  final A-w SDR : " << final_perc_sdr_db << " dB\n";
  out_ << "  elapsed       : " << elapsed_s << " s ("
       << (static_cast<double>(frames_done) / std::max(elapsed_s, 1e-6))
       << " fps)\n";
  if (abr) {
    out_ << "  ABR target    : " << std::setprecision(0) << abr->target_kbps
         << " kbps\n";
    out_ << "  ABR achieved  : " << std::setprecision(2) << abr->avg_kbps
         << " kbps (" << (abr->avg_kbps - abr->target_kbps) << ")\n";
  }
  out_ << G
       << "--------------------------------------------------------------------"
          "--\n"
       << "  done\n"
       << "===================================================================="
          "==\n"
       << R;
  out_.flush();
}

} // namespace rmps::tui
