#include "control/transient_gate.hpp"
#include "core/encoder_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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
/// HF cut applied to an attack granule and (below) its run-up. Both are far
/// above the Mid lowpass at 48-64 kbps once iteration_loop scales them (x1.4
/// on Mid), so by default neither reaches the band where the pre-echo is.
/// RMPS_TRANSIENT_CUT_HZ / RMPS_PRE_WINDOW_CUT_HZ make them measurable.
inline double transient_cut_hz() {
  static const double v = [] {
    if (const char *e = std::getenv("RMPS_TRANSIENT_CUT_HZ")) {
      const double x = std::atof(e);
      if (x > 1000.0) return x;
    }
    return 18000.0;
  }();
  return v;
}
inline double pre_window_cut_hz() {
  static const double v = [] {
    if (const char *e = std::getenv("RMPS_PRE_WINDOW_CUT_HZ")) {
      const double x = std::atof(e);
      if (x > 1000.0) return x;
    }
    return 13000.0;
  }();
  return v;
}
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

/// One pass over a sub-window, for one derived signal.
///
/// `pick` maps a sample index to the scalar being measured. Stereo is measured
/// twice — once on Mid, once on Side — and the two results combined by
/// `measure_subwindow` below.
template <typename Pick>
SubWindowMetrics measure_one(int len, Pick&& pick) {
  auto mid = [&](int idx) { return pick(idx); };
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

/// Sub-window metrics for a stereo (or mono) frame.
///
/// Stereo used to be collapsed to `mid = (l + r) / 2` and measured once. That
/// makes the detector blind to anything the two channels disagree about: an
/// anti-correlated attack — a hard-panned hi-hat against something on the other
/// side, a wide reverb, anything through a stereo widener — cancels in the sum
/// and simply is not there to detect. A transient the gate cannot see gets no
/// short block and no bit boost, which is the pre-echo case the whole gate
/// exists for.
///
/// Measuring Mid *and* Side and taking the larger of each metric fixes that
/// without recalibrating anything. Every test downstream is a ratio — energy
/// against the previous sub-window, peak against RMS, HF against LF — so a
/// change of scale does not move a threshold; only cancellation does. For
/// centred content Side is ~0 and Mid is the old signal, so the numbers are
/// the ones the thresholds were tuned on; for hard-panned content Mid and Side
/// each carry half, exactly as `(l + r) / 2` did. Only the anti-correlated case
/// changes, and there it changes from "invisible" to "seen".
///
/// The HF/LF ratio is the exception: it stays on Mid alone.
///
/// The two measurements answer different questions. Energy, peak and RMS are
/// used to detect *that* an attack happened, and there Side has to be looked at
/// or an anti-correlated attack is invisible. The HF/LF ratio is used to
/// classify *what kind* of attack it is — `short_block_hf_threshold` asks
/// whether it is a hi-hat rather than a kick — and the Side channel of ordinary
/// stereo music is decorrelated reverb and stereo width, which is HF-rich all
/// the time, attack or not. Scoring the classifier on Side therefore says "HF
/// transient" almost everywhere.
///
/// Measured: taking the HF/LF ratio from Side as well raised short-block usage
/// from 1.70 % to 8.36 % of granules on `witchqueen` and from 1.22 % to 4.44 %
/// on `avicii`, and the harness rejected it outright — `mixed_music` segSNR
/// 31.65 -> 19.43 dB at 48 kbps, segSNR_p05 worse at every bitrate from 48 to
/// 192, with `pre_echo_db` not moving at all. All of the frequency resolution
/// the short blocks cost, none of the pre-echo they are for.
SubWindowMetrics measure_subwindow(const std::int16_t *pcm, int start, int len,
                                   int interleave, int channels) {
  static const bool mid_only = [] {
    const char* e = std::getenv("RMPS_GATE_MID_ONLY");
    return e && std::atoi(e) != 0;
  }();
  if (channels != 2) {
    return measure_one(len, [&](int idx) {
      return static_cast<int>(pcm[(start + idx) * interleave]);
    });
  }
  const SubWindowMetrics mid = measure_one(len, [&](int idx) {
    const int l = pcm[(start + idx) * interleave + 0];
    const int r = pcm[(start + idx) * interleave + 1];
    return (l + r) / 2;
  });
  if (mid_only) return mid;
  const SubWindowMetrics side = measure_one(len, [&](int idx) {
    const int l = pcm[(start + idx) * interleave + 0];
    const int r = pcm[(start + idx) * interleave + 1];
    return (l - r) / 2;
  });
  SubWindowMetrics m{};
  m.energy = std::max(mid.energy, side.energy);
  m.peak   = std::max(mid.peak, side.peak);
  m.rms    = std::max(mid.rms, side.rms);
  m.hf_energy = mid.hf_energy;   // classifier: Mid only, see above
  m.lf_energy = mid.lf_energy;
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
      static const bool hf_debug = (std::getenv("RMPS_HF_RATIO") != nullptr);
      if (hf_debug)
        std::fprintf(stderr, "[hfr] %.5f\n", hf_to_lf);
    }
    decisions_[gr].cutoff_hz = any_from_silence ? ATTACK_FROM_SILENCE_CUTOFF_HZ
                               : any            ? transient_cut_hz()
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
          std::min(decisions_[gr - 1].cutoff_hz, pre_window_cut_hz());
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

/// Cross-frame START placement (below). **Default off — it measured worse.**
///
/// The idea is sound and the defect it targets is real: an attack in granule 0
/// needs its START window in the *previous* frame, which the gate cannot see,
/// so half of all attacks get their short block placed a granule late. The
/// one-frame PCM lookahead can supply that. Three versions were measured:
///
///   * raw energy step        20 eval cells worse, mixed_music@48 29.5 -> 9.9 dB.
///                            `probe_demand`'s own comment says it "steers a bit
///                            grant, not a coding decision" — it fires on
///                            ordinary music.
///   * + the gate's HF test   8 worse / 2 better. Better, still negative.
///   * + polyphase delay      neutral: segSNR 0/0, hole rate 2 cells slightly
///     correction (the ~481   worse. It *is* live — transient_train gets 62 %
///     sample analysis delay  more short granules (16 -> 26) — but pre_echo
///     means the incoming     goes 0.19-0.36 dB the wrong way at 48/64/96.
///     PCM's first half lands
///     in the *second* granule)
///
/// So more short blocks, correctly placed, measured as slightly *more*
/// pre-echo. The likely reason is that the START window this buys is itself
/// poorly time-confined, and paying for one on the preceding granule costs
/// more than putting the SHORT on the attack gains. Enable with
/// RMPS_LA_SHORT=1 if revisiting; `RMPS_LA_HALF` selects which half of the
/// probe window counts as granule 0.
static bool la_short_disabled() {
  static const bool off = [] {
    const char *e = std::getenv("RMPS_LA_SHORT");
    return !(e && std::atoi(e) != 0);
  }();
  return off;
}

void decide_block_types(EncoderState& s) {
  const int G = s.mpeg.granules_per_frame;
  for (int gr = 0; gr < G; ++gr) {
    // Desired block from the PCM-domain transient analysis. Only HF-rich
    // attacks (hi-hats, snares, broadband clicks) warrant a short block; a
    // low-frequency transient (kick, bass-note onset) stays long so its bass
    // keeps full frequency resolution. pre_window marks the granule before an
    // attack — the natural place for the START transition so the SHORT lands on
    // the attack itself, but only when that next attack is itself HF-rich.
    int desired = BLOCK_LONG;
    if (s.cfg_block_switching) {
      if (s.transient_flags[gr] && s.hf_transient_flags[gr]) {
        desired = BLOCK_SHORT;
      } else if (s.pre_window_flags[gr] && gr + 1 < G &&
                 s.transient_flags[gr + 1] && s.hf_transient_flags[gr + 1]) {
        desired = BLOCK_START;
      } else if (gr == G - 1 && s.lookahead_attack_gr0 && !la_short_disabled()) {
        // The attack lands in the NEXT frame's granule 0.
        //
        // Without this the short block is placed a granule late, and for half
        // of all attacks it cannot be placed correctly at all. The state
        // machine below turns "I want SHORT" at a granule whose predecessor is
        // LONG into a START *at that granule*, with the SHORT falling on the
        // one after — so the attack granule itself gets a long-ish window and
        // the short window covers the decay. The `pre_window` branch above
        // exists to place the START one granule early instead, but its
        // condition is `gr + 1 < G` and G is 2, so it can only anticipate an
        // attack in granule 1. An attack in granule 0 needs a START in the
        // *previous frame*, which was already committed by the time the gate
        // sees it.
        //
        // The one-frame PCM lookahead already in the encoder supplies exactly
        // that: a stateless probe reports whether the incoming frame's first
        // half carries the energy step. Emitting START here lets the next
        // frame's granule 0 be forced to SHORT by the transition rules, which
        // puts the short window on the attack itself.
        desired = BLOCK_START;
        // A START here is *forced* by the window-switching rules, not asked
        // for by this granule's own content, and it costs: measured, START
        // spreads a coding error over the full 36-sample span exactly like
        // LONG (see the window analysis) while having the shorter effective
        // length — i.e. worse frequency resolution — of a transition window.
        // This granule is the run-up before the attack, which is precisely
        // where pre-echo is measured, so degrading it there is what made the
        // first version of this feature come out *worse* on pre_echo despite
        // placing the short block correctly.
        //
        // Mark it as a pre-window so the machinery that already exists for
        // run-up granules (lower masking thresholds via
        // `pre_mask_transient`, TRANSIENT_WEIGHT in the band cost) protects
        // it. Measured inert — the granules this fires on already carried the
        // flag from the gate's own run-up detection — but it is the correct
        // statement about a granule the switching rules forced, so it stays.
        s.pre_window_flags[gr] = true;
      }
    }

    // Reconcile against the previous granule's block type so the emitted
    // sequence is always a legal window-switching transition:
    //   LONG  -> LONG | START          STOP  -> LONG | START
    //   START -> SHORT (forced)        SHORT -> SHORT | STOP
    const int prev = s.prev_block_type[0];
    int block;
    if (prev == BLOCK_START) {
      block = BLOCK_SHORT;                                   // start must precede short
    } else if (prev == BLOCK_SHORT) {
      block = (desired == BLOCK_LONG) ? BLOCK_STOP : BLOCK_SHORT;
    } else if (prev == BLOCK_STOP) {
      block = (desired == BLOCK_LONG) ? BLOCK_LONG : BLOCK_START;
    } else { // prev == BLOCK_LONG
      block = (desired == BLOCK_LONG) ? BLOCK_LONG : BLOCK_START;
    }

    for (int ch = 0; ch < MAX_CHANNELS; ++ch) {
      s.block_type[ch][gr] = block;
      s.prev_block_type[ch] = block;
    }
  }
}

} // namespace rmps::control
