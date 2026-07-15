#include "control/transient_gate.hpp"
#include "core/encoder_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace rmps::control {

namespace {

// 192 PCM samples covers ~4.35 ms at 44.1 kHz -- matches the short-block
// time resolution. We split each granule's 576 PCM samples into three 192-
// sample sub-windows.
constexpr int SUBWINDOW = 192;
constexpr int SUBWINDOWS_PER_GR = 3;

// Tunable thresholds.
constexpr double ENERGY_JUMP_TRIG = 3.0; // hard trigger
constexpr double ENERGY_SUSTAINED_TRIG =
    2.0; // softer trigger if accompanied by peak
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
constexpr double TRANSIENT_CUTOFF_HZ = 18000.0;
constexpr double PRE_WINDOW_CUTOFF_HZ = 13000.0;
constexpr double NO_TRANSIENT_CUTOFF_HZ = 20000.0;
constexpr double ATTACK_FROM_SILENCE_CUTOFF_HZ = 15000.0;

// Below this sub-window energy we consider the region "silent" for the
// purpose of detecting attacks-from-silence. 1e5 over 192 samples ≈ 22 RMS
// in 16-bit, well under any musical content but above quiet dither.
constexpr double SILENCE_ENERGY_THRESHOLD = 1.0e5;

// Length of the boxcar low-pass used to isolate the bass / low-mid band when
// classifying a transient. At 48 kHz a 16-tap moving average has its first null
// at 3 kHz (−3 dB ≈ 1.3 kHz), so its output energy tracks the content below
// ~1 kHz — i.e. the bass a short block would damage.
constexpr int LF_BOXCAR = 16;

struct SubWindowMetrics {
  double energy;
  double peak;
  double rms;
  /// First-difference (high-pass) energy — large for a hi-hat / snap, tiny for
  /// a bass thump.
  double hf_energy;
  /// Boxcar-low-pass energy — the bass / low-mid content a short block would
  /// damage. We classify an attack as short-block-worthy only when its HF
  /// energy clearly dominates this, so a broadband hit that still carries
  /// heavy bass (kick, bass-note onset, snare-with-body) stays on long blocks.
  double lf_energy;
};

SubWindowMetrics measure_subwindow(const std::int16_t *pcm, int start, int len,
                                   int interleave, int channels) {
  auto mid = [&](int idx) {
    if (channels == 2) {
      const int l = pcm[(start + idx) * interleave + 0];
      const int r = pcm[(start + idx) * interleave + 1];
      return (l + r) / 2;
    }
    return static_cast<int>(pcm[(start + idx) * interleave]);
  };
  double e = 0.0;
  double hf = 0.0;
  double lf = 0.0;
  int pk = 0;
  int prev = 0;
  double box = 0.0; // running sum of the last LF_BOXCAR samples
  for (int i = 0; i < len; ++i) {
    const int s = mid(i);
    const int a = std::abs(s);
    if (a > pk)
      pk = a;
    e += static_cast<double>(s) * static_cast<double>(s);
    if (i > 0) {
      const double d = static_cast<double>(s - prev);
      hf += d * d;
    }
    prev = s;
    box += s;
    if (i >= LF_BOXCAR)
      box -= mid(i - LF_BOXCAR);
    if (i >= LF_BOXCAR - 1) {
      const double avg = box / LF_BOXCAR;
      lf += avg * avg;
    }
  }
  SubWindowMetrics m{};
  m.energy = std::max(e, 1.0);
  m.peak = pk;
  m.rms = std::sqrt(m.energy / std::max(len, 1));
  m.hf_energy = hf;
  m.lf_energy = std::max(lf, 1.0);
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
    int first_attack_sub = -1;      // earliest attacking sub-window
    double best_contrast = 1.0;     // strongest energy jump
    for (int w = 0; w < SUBWINDOWS_PER_GR; ++w) {
      const int s = gr * SUBWINDOWS_PER_GR + w;
      // Store per-sub-window energy in the decision.
      decisions_[gr].sub_energies[w] = per_sub[s].energy;
      if (trig[s]) {
        any = true;
        if (from_silence[s])
          any_from_silence = true;
        if (pk_ratio[s] > best_ratio)
          best_ratio = pk_ratio[s];
        if (first_attack_sub < 0)
          first_attack_sub = w;
        // Energy contrast: ratio of this sub to the previous sub.
        double prev_sub_e = 1.0;
        if (s > 0)
          prev_sub_e = std::max(per_sub[s - 1].energy, 1.0);
        else if (!cold_start)
          prev_sub_e = std::max(last_window_energy_, 1.0);
        const double contrast = per_sub[s].energy / prev_sub_e;
        if (contrast > best_contrast)
          best_contrast = contrast;
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
    decisions_[gr].attack_sub_index = first_attack_sub;
    decisions_[gr].energy_contrast = best_contrast;
    // Classify the attack as HF-rich (hi-hat/snare → short block worthwhile) vs
    // a low-frequency thump (kick/bass onset → keep long for clean bass) from
    // the first-difference energy fraction of the attacking sub-window.
    if (any && first_attack_sub >= 0) {
      const auto &am = per_sub[gr * SUBWINDOWS_PER_GR + first_attack_sub];
      // Compare the attack's high-frequency energy directly against its bass /
      // low-mid energy. A short block is only worth its bass-resolution cost
      // when the HF clearly dominates (hi-hat, snare snap, cymbal); any attack
      // whose energy is still carried by the low end — a kick, a bass-note
      // onset, or a snare/clap with body — stays long so the bass keeps full
      // frequency resolution. This ratio test rejects broadband bass-heavy
      // hits that a plain HF-fraction test let through.
      const double hf_to_lf = am.hf_energy / am.lf_energy;
      decisions_[gr].hf_transient = hf_to_lf >= hf_transient_threshold_;
    }
    decisions_[gr].cutoff_hz = NO_TRANSIENT_CUTOFF_HZ;
    if (any)
      ++transient_count_;
  }
  // Mark the granule before a transient as a pre-echo zone so we suppress
  // HF there too (this is the silence-before-attack region).
  // We no longer reduce the hard LPF cutoff_hz here because the dynamic
  // jumping of the high-frequency cutoff (e.g. 20k -> 13k -> 18k -> 20k)
  // creates frame-boundary modulation artifacts, which decode as low-frequency
  // "boff" thumps. We let the masking model's pre-echo penalty handle the
  // bit distribution instead.
  for (int gr = granules_per_frame - 1; gr >= 1; --gr) {
    if (decisions_[gr].transient && !decisions_[gr - 1].transient) {
      decisions_[gr - 1].pre_window = true;
      decisions_[gr - 1].cutoff_hz = NO_TRANSIENT_CUTOFF_HZ;
    }
  }

  // Cross-frame pre_window: if the first granule of this frame is a
  // transient, the *last* granule of the *previous* frame should have been
  // flagged as pre_window. We cannot retroactively change the already-
  // encoded frame, but the encoder can use prev_frame_last_decision() to
  // apply the pre_window masking penalty to the masking history, which
  // carries forward into this frame's thresholds via the post-mask decay.
  // (The actual application happens in encoder.cpp run_frame.)
  //
  // Save a snapshot of the current frame's last granule *before* overwrite.
  prev_frame_last_decision_ = decisions_[granules_per_frame - 1];
}

TransientGate::Decision TransientGate::decision_for(int gr) const noexcept {
  if (gr < 0 || gr >= static_cast<int>(std::size(decisions_)))
    return {};
  return decisions_[gr];
}

void decide_block_types(EncoderState& s) {
  const int G = s.mpeg.granules_per_frame;
  int desired[2] = {BLOCK_LONG, BLOCK_LONG};

  if (s.cfg_block_switching) {
    for (int gr = 0; gr < G; ++gr) {
      if (s.transient_flags[gr] && s.hf_transient_flags[gr]) {
        desired[gr] = BLOCK_SHORT;
      }
    }
  }

  int prev = s.prev_block_type[0];

  for (int gr = 0; gr < G; ++gr) {
    int block = BLOCK_LONG;

    if (desired[gr] == BLOCK_SHORT) {
      if (prev == BLOCK_LONG || prev == BLOCK_STOP) {
        // If the attack is in this granule but the previous granule was LONG/STOP,
        // we couldn't place the START window beforehand due to look-ahead limits.
        // Forcing a transition here would make this granule START, shifting the
        // SHORT block by 1 granule and corrupting the transient onset (causing "boff" thumps).
        // Safely fallback to BLOCK_LONG to let the cost model handle it.
        block = BLOCK_LONG;
      } else if (prev == BLOCK_START) {
        block = BLOCK_SHORT;
      } else if (prev == BLOCK_SHORT) {
        block = BLOCK_SHORT;
      }
    } else { // desired[gr] == BLOCK_LONG
      if (prev == BLOCK_START) {
        // MPEG standard constraint: START must be followed by SHORT
        block = BLOCK_SHORT;
      } else if (prev == BLOCK_SHORT) {
        block = BLOCK_STOP;
      } else if (prev == BLOCK_STOP) {
        block = BLOCK_LONG;
      } else {
        block = BLOCK_LONG;
      }
    }

    // Look-ahead within the current frame: if gr=0 is LONG and gr=1 is SHORT,
    // we can place START in gr=0 legally.
    if (gr == 0 && block == BLOCK_LONG && desired[1] == BLOCK_SHORT) {
      block = BLOCK_START;
    }

    prev = block;
    for (int ch = 0; ch < MAX_CHANNELS; ++ch) {
      s.block_type[ch][gr] = block;
      s.prev_block_type[ch] = block;
    }
  }
}

} // namespace rmps::control
