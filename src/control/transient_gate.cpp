#include "control/transient_gate.hpp"

#include <algorithm>
#include <cmath>

namespace rmps::control {

namespace {

// 192 PCM samples covers ~4.35 ms at 44.1 kHz -- matches the short-block
// time resolution. We split each granule's 576 PCM samples into three 192-
// sample sub-windows.
constexpr int SUBWINDOW = 192;
constexpr int SUBWINDOWS_PER_GR = 3;

// Tunable thresholds.
constexpr double ENERGY_JUMP_TRIG = 4.0; // hard trigger
constexpr double ENERGY_SUSTAINED_TRIG =
    2.5; // softer trigger if accompanied by peak
constexpr double PEAK_OVER_RMS_TRIG =
    2.0; // any peak that breaks above sustained RMS
constexpr double ABS_ENERGY_FLOOR =
    1e7; // ignore micro-energy ratios in silence

// HF cutoffs applied to transient granules. Hi-hats and cymbals carry
// energy up to ~22 kHz, and the old cutoffs (5–14 kHz) were specifically
// breaking that "sizzle". ABR now feeds attack frames up to 320 kbps
// and the L2 tonality discount is disabled on transient granules, so
// the HF can be encoded properly — no need for a hard gate.
//
// We keep a very gentle trim on attack-from-silence only, to suppress
// the sharpest of the long-block pre-echo smear. Everything else passes
// through and the cost function decides.
constexpr double TRANSIENT_CUTOFF_HZ = 20000.0;
constexpr double PRE_WINDOW_CUTOFF_HZ = 20000.0;
constexpr double NO_TRANSIENT_CUTOFF_HZ = 20000.0;
constexpr double ATTACK_FROM_SILENCE_CUTOFF_HZ = 16000.0;

// Below this sub-window energy we consider the region "silent" for the
// purpose of detecting attacks-from-silence. 1e5 over 192 samples ≈ 22 RMS
// in 16-bit, well under any musical content but above quiet dither.
constexpr double SILENCE_ENERGY_THRESHOLD = 1.0e5;

struct SubWindowMetrics {
  double energy;
  double peak;
  double rms;
};

SubWindowMetrics measure_subwindow(const std::int16_t *pcm, int start, int len,
                                   int interleave, int channels) {
  double e = 0.0;
  int pk = 0;
  for (int i = 0; i < len; ++i) {
    // Use mid (L+R)/2 to be channel-agnostic without quirks.
    int s;
    if (channels == 2) {
      const int l = pcm[(start + i) * interleave + 0];
      const int r = pcm[(start + i) * interleave + 1];
      s = (l + r) / 2;
    } else {
      s = pcm[(start + i) * interleave];
    }
    const int a = std::abs(s);
    if (a > pk)
      pk = a;
    e += static_cast<double>(s) * static_cast<double>(s);
  }
  SubWindowMetrics m{};
  m.energy = std::max(e, 1.0);
  m.peak = pk;
  m.rms = std::sqrt(m.energy / std::max(len, 1));
  return m;
}

} // namespace

void TransientGate::analyse_frame(const std::int16_t *pcm,
                                  std::size_t samples_per_pass, int channels,
                                  int granules_per_frame) {
  transient_count_ = 0;
  for (auto &d : decisions_)
    d = Decision{};

  if (!pcm || samples_per_pass == 0 || channels <= 0)
    return;

  const int interleave = channels;

  // Measure energy per granule's 3 sub-windows and decide per granule.
  SubWindowMetrics per_sub[2 * SUBWINDOWS_PER_GR]{};
  int total_subs = 0;
  for (int gr = 0; gr < granules_per_frame; ++gr) {
    for (int w = 0; w < SUBWINDOWS_PER_GR; ++w) {
      const int start =
          gr * static_cast<int>(samples_per_pass / granules_per_frame) +
          w * SUBWINDOW;
      per_sub[total_subs] =
          measure_subwindow(pcm, start, SUBWINDOW, interleave, channels);
      ++total_subs;
    }
  }

  // Now walk sub-windows in order and look for energy jumps vs the
  // *previous* sub-window (including across granule boundaries) and a
  // sharp peak/RMS signature in the offending sub-window.
  //
  // Cold start (last_window_energy_ < 0): nothing has played yet, so
  // semantically the "previous sub-window" is silence — exactly the
  // condition we most want to detect for the song-starts-from-silence
  // case. Bootstrap prev_e to the silence floor (1.0) instead of the
  // current sub-window's own energy, otherwise the very first attack
  // compares to itself and ratio_e collapses to 1.0.
  const bool cold_start = last_window_energy_ < 0.0;
  double prev_e = cold_start ? 1.0 : last_window_energy_;
  bool trig[2 * SUBWINDOWS_PER_GR]{};
  bool from_silence[2 * SUBWINDOWS_PER_GR]{};
  double pk_ratio[2 * SUBWINDOWS_PER_GR]{};
  for (int s = 0; s < total_subs; ++s) {
    const auto &m = per_sub[s];
    const double ratio_e = m.energy / std::max(prev_e, 1.0);
    const double ratio_p = m.peak / std::max(m.rms, 1.0);
    pk_ratio[s] = ratio_p;
    const bool prev_was_silent = prev_e < SILENCE_ENERGY_THRESHOLD;
    bool trans = false;
    if (m.energy > ABS_ENERGY_FLOOR) {
      if (ratio_e > ENERGY_JUMP_TRIG)
        trans = true;
      else if (ratio_e > ENERGY_SUSTAINED_TRIG && ratio_p > PEAK_OVER_RMS_TRIG)
        trans = true;
      // Independent of ratio: any musically-significant sub-window arriving
      // straight out of silence is a silence→attack event by definition.
      // The ratio-based test already catches this when prev is at the floor
      // (1.0), but this branch covers the boundary case where prev was
      // "quiet but not floor-clamped" (e.g. tail of a reverb, noise gate).
      if (prev_was_silent)
        trans = true;
    }
    trig[s] = trans;
    from_silence[s] = trans && prev_was_silent;
    prev_e = m.energy;
  }
  last_window_energy_ = per_sub[total_subs - 1].energy;

  // Roll up to per-granule decision.
  for (int gr = 0; gr < granules_per_frame; ++gr) {
    bool any = false;
    bool any_from_silence = false;
    bool all_silent = true;
    double best_ratio = 0.0;
    for (int w = 0; w < SUBWINDOWS_PER_GR; ++w) {
      const int s = gr * SUBWINDOWS_PER_GR + w;
      if (trig[s]) {
        any = true;
        if (from_silence[s])
          any_from_silence = true;
        if (pk_ratio[s] > best_ratio)
          best_ratio = pk_ratio[s];
      }
      if (per_sub[s].energy >= SILENCE_ENERGY_THRESHOLD)
        all_silent = false;
    }
    decisions_[gr].transient = any;
    decisions_[gr].attack_from_silence = any_from_silence;
    // A granule containing an attack obviously isn't "silent", even if its
    // first sub-window is.
    decisions_[gr].silent = all_silent && !any;
    decisions_[gr].peak_rms_ratio = best_ratio;
    decisions_[gr].cutoff_hz = any_from_silence ? ATTACK_FROM_SILENCE_CUTOFF_HZ
                               : any            ? TRANSIENT_CUTOFF_HZ
                                                : NO_TRANSIENT_CUTOFF_HZ;
    if (any)
      ++transient_count_;
  }
  // Mark the granule before a transient as a pre-echo zone so we suppress
  // HF there too (this is the silence-before-attack region).
  for (int gr = granules_per_frame - 1; gr >= 1; --gr) {
    if (decisions_[gr].transient && !decisions_[gr - 1].transient) {
      decisions_[gr - 1].pre_window = true;
      decisions_[gr - 1].cutoff_hz =
          std::min(decisions_[gr - 1].cutoff_hz, PRE_WINDOW_CUTOFF_HZ);
    }
  }
}

TransientGate::Decision TransientGate::decision_for(int gr) const noexcept {
  if (gr < 0 || gr >= static_cast<int>(std::size(decisions_)))
    return {};
  return decisions_[gr];
}

} // namespace rmps::control
