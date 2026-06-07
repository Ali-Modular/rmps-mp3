#include "post/lowpass.hpp"
#include "core/encoder_state.hpp"

#include <algorithm>

namespace rmps::post {

namespace {

constexpr double MID_CUTOFF_HZ = 20000.0;

struct Step {
  int min_kbps;
  double cutoff_hz;
};
constexpr Step LPF_SCHEDULE[] = {
    {320, 20000.0}, {256, 19500.0}, {192, 19000.0}, {160, 18000.0},
    {128, 17000.0}, {112, 15000.0}, {96, 14000.0},  {80, 12000.0},
    {64, 10000.0},  {56, 8000.0},   {48, 7000.0},   {40, 6000.0},
    {0, 5000.0},
};

int hz_to_bin(double hz, int samplerate) {
  int n = static_cast<int>(hz * 576.0 * 2.0 / samplerate);
  return std::clamp(n, 0, GRANULE_SAMPLES);
}

} // namespace

double lowpass_side_cutoff_hz(int bitr_kbps) {
  for (const auto &s : LPF_SCHEDULE)
    if (bitr_kbps >= s.min_kbps)
      return s.cutoff_hz;
  return LPF_SCHEDULE[std::size(LPF_SCHEDULE) - 1].cutoff_hz;
}

void apply_lowpass(EncoderState &s) {
  const double nyq = s.wave.samplerate * 0.5;
  const double m_hz = std::min(MID_CUTOFF_HZ, nyq - 1.0);
  const double s_hz = std::min(lowpass_side_cutoff_hz(s.mpeg.bitr), nyq - 1.0);
  const int m_bin = hz_to_bin(m_hz, s.wave.samplerate);
  const int s_bin = hz_to_bin(s_hz, s.wave.samplerate);
  const bool is_ms = (s.wave.channels == 2 && (s.mpeg.mode_ext & 2));

  for (int ch = 0; ch < s.wave.channels; ++ch) {
    const int cutoff = is_ms ? (ch == 0 ? m_bin : s_bin) : s_bin;
    if (cutoff >= GRANULE_SAMPLES)
      continue;
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
      auto &arr = s.mdct_freq[ch][gr];
      for (int i = cutoff; i < GRANULE_SAMPLES; ++i)
        arr[i] = 0;
    }
  }
}

} // namespace rmps::post
