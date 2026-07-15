#include "post/lowpass.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"

#include <algorithm>
#include <cmath>

namespace rmps::post {

namespace {

constexpr double MID_CUTOFF_HZ = 20000.0;

struct Step {
  int min_kbps;
  double cutoff_hz;
};
// Side-channel cutoff. Below ~160 kbps the Side channel is the cheapest
// place to save bits: its HF is far less audible than Mid HF, and every
// line it stops coding flows into Mid/bass precision via ms_xfer and the
// frame pool. Mid keeps a wider band at every rate (see
// lowpass_mid_cutoff_hz below).
constexpr Step LPF_SCHEDULE[] = {
    {160, 20000.0}, {128, 17000.0}, {112, 15000.0}, {96, 13000.0},
    {80, 11500.0},  {64, 10000.0},  {48, 8000.0},
    {0, 7000.0},
};

int hz_to_bin(double hz, int samplerate) {
  int n = static_cast<int>(hz * 576.0 * 2.0 / samplerate);
  return std::clamp(n, 0, GRANULE_SAMPLES);
}

double lowpass_mid_cutoff_hz(int bitr_kbps) {
  if (bitr_kbps >= 64)
    return 20000.0;
  if (bitr_kbps >= 48)
    return 14000.0;
  return 12000.0;  // 32-40 kbps (the old fallthrough returned 20 kHz here)
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
  const double m_hz = std::min(lowpass_mid_cutoff_hz(s.mpeg.bitr), nyq - 1.0);
  const double s_hz = std::min(lowpass_side_cutoff_hz(s.mpeg.bitr), nyq - 1.0);
  const int m_bin = hz_to_bin(m_hz, s.wave.samplerate);
  const int s_bin = hz_to_bin(s_hz, s.wave.samplerate);
  const bool is_ms = (s.wave.channels == 2 && (s.mpeg.mode_ext & 2));

  for (int ch = 0; ch < s.wave.channels; ++ch) {
    const int cutoff = is_ms ? (ch == 0 ? m_bin : s_bin) : m_bin;
    if (cutoff >= GRANULE_SAMPLES)
      continue;
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
      auto &arr = s.mdct_freq[ch][gr];
      for (int i = cutoff; i < GRANULE_SAMPLES; ++i)
        arr[i] = 0;
    }
  }
}

void apply_hf_gain_boost(EncoderState &s) {
  if (s.cfg_hf_gain_boost_14k <= 1.0)
    return;

  const int bin_clamp = hz_to_bin(s.frame_clamp_hz, s.wave.samplerate);
  if (bin_clamp >= GRANULE_SAMPLES)
    return;

  for (int ch = 0; ch < s.wave.channels; ++ch) {
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
      auto &arr = s.mdct_freq[ch][gr];
      for (int i = bin_clamp; i < GRANULE_SAMPLES; ++i) {
        arr[i] = fx::i32(static_cast<std::int64_t>(arr[i] * s.cfg_hf_gain_boost_14k));
      }
    }
  }
}

} // namespace rmps::post
