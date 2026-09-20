#include "post/lowpass.hpp"
#include "core/encoder_state.hpp"
#include "core/tables.hpp"
#include "post/psy_common.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace rmps::post {

namespace {

constexpr double MID_CUTOFF_HZ = 20000.0;

int hz_to_bin(double hz, int samplerate) {
  int n = static_cast<int>(hz * 576.0 * 2.0 / samplerate);
  return std::clamp(n, 0, GRANULE_SAMPLES);
}

/// Lowest cut this encoder will ever apply to Mid, by rate.
///
/// This used to be a flat 13800 Hz, with the comment that below it the loss is
/// a timbre change rather than a bit saving worth having. That was never
/// measured at a starved rate, and it is wrong there — worse, it was the
/// binding constraint: the scan below starts at this value and `continue`s past
/// every band under it, so the "adaptive" cutoff could only ever choose between
/// 13.8 and 20 kHz, at 32 kbps exactly as at 320.
///
/// Measured on real program material at 48 kbps (`scratchpad/harmcont.py`, per
/// band, 30 s excerpts), 10-16 kHz sat at a bias of -52.6 dB — 100 % of its
/// sustained partials dropped, paid for and not delivered — while 3-6 kHz lost
/// 30 % of its partials. Pinning the cut with `RMPS_LPF`:
///
///     LPF     1-3k drop   3-6k drop   3-6k bias   6-10k bias
///     auto      8.4 %      29.4 %      -6.25       -25.69
///     12k       4.2 %      23.0 %      -4.83       -22.88
///     10k       2.8 %      20.6 %      -3.84       -21.11
///     8.5k      1.8 %      16.0 %      -2.95       -33.93   <- past the knee
///
/// 6-10 kHz *improves* as well until the cut starts eating it, so this is not a
/// brightness-for-midrange trade — it is the same bits over fewer lines.
double lpf_floor_hz(const EncoderState& s) {
  const int br = s.policy_bitr();
  if (br <= 0 || br >= 112) return 13800.0;
  if (br >= 96)  return 13000.0;
  if (br >= 80)  return 12000.0;
  if (br >= 64)  return 11000.0;
  if (br >= 48)  return 10000.0;
  return 9000.0;
}

/// How far under the absolute threshold of hearing a band may sit and still be
/// considered worth coding. 0 dB is the definition of inaudible; a few dB of
/// margin keeps the decision off a knife edge for content that hovers there.
///
/// Making this rate-dependent was tried — negative at starved rates, so a band
/// must clear ATH by some amount to earn its bits — on the theory that it would
/// be a gentler, material-adaptive version of the rate cap below. It is
/// **inert on real material**: at 48 kbps a -10 dB requirement moved kk30's
/// 3-6 kHz dropout from 29.4 % to 29.5 %, i.e. not at all, while a 12 kHz pin
/// gave 23.0 %. The reason is that the problem is not audibility. Real music's
/// 10-14 kHz sits far above ATH — it would be plainly audible on its own — it
/// simply cannot be *afforded* at 48 kbps, and comes back at -52 dB. ATH cannot
/// express affordability, so the cap below has to, and stays a cap.
constexpr double LPF_ATH_MARGIN_DB = 6.0;

/// Adaptive Mid cutoff: keep everything the ear could actually hear, cut the
/// rest.
///
/// The schedule this replaces was a flat 20 kHz at every bitrate — no lowpass
/// at all. Coding to 20 kHz on a starved budget spreads the bits thin enough
/// that the whole spectrum comes back coarse, and the broadband error shows up
/// in the time domain as peaks over the source (measured at 64 kbps: 0.061 %
/// of samples above the source peak, worst +5.3 dB). A fixed rate-indexed
/// schedule fixes that but throws away real content on material that has it.
///
/// So ask the signal instead. Walk the band table from the floor upward and
/// keep the highest band whose own level clears the absolute threshold of
/// hearing; everything above that is inaudible by definition and its bits are
/// better spent below. `ath_db_at` needs no masking thresholds, only energy
/// and frequency, so this runs before the psychoacoustic model without any
/// ordering problem.
/// Adaptive Mid cutoff: keep everything the ear could actually hear, cut the
/// rest.
///
/// The schedule this replaces was a flat 20 kHz at every bitrate — no lowpass
/// at all. Coding to 20 kHz on a starved budget spreads the bits thin enough
/// that the whole spectrum comes back coarse, and the broadband error shows up
/// in the time domain as peaks over the source (measured at 64 kbps: 0.061 %
/// of samples above the source peak, worst +5.3 dB). A fixed rate-indexed
/// schedule fixes that but throws away real content on material that has it.
///
/// So ask the signal instead. Walk the band table from the floor upward and
/// keep the highest band whose own level clears the absolute threshold of
/// hearing; everything above that is inaudible by definition and its bits are
/// better spent below. `ath_db_at` needs no masking thresholds, only energy
/// and frequency, so this runs before the psychoacoustic model without any
/// ordering problem.
/// Rate cap on the Mid cutoff: "is this band worth its bits here?"
///
/// The scan answers a different question — "is this band audible at all" — and
/// it had no rate term whatsoever, so it returned the same cutoff at 48 and at
/// 320 kbps. Side always had `side_cut_fraction(bitr)`; Mid never did.
///
/// Measured on real program material (`scratchpad/harmcont.py`, per band, 30 s
/// excerpts). At 48 kbps 10-16 kHz sat at a bias of **-52.6 dB** — 100 % of its
/// sustained partials dropped, paid for and not delivered — while 3-6 kHz, the
/// most audible band there is, lost 30 % of its partials. Pinning the cut:
///
///     LPF     1-3k drop   3-6k drop   3-6k bias   6-10k bias
///     auto      8.4 %      29.4 %      -6.25       -25.69
///     12k       4.2 %      23.0 %      -4.83       -22.88
///     10k       2.8 %      20.6 %      -3.84       -21.11
///     8.5k      1.8 %      16.0 %      -2.95       -33.93   <- past the knee
///
/// 6-10 kHz *improves* too until the cut starts eating it, so this is not a
/// brightness-for-midrange trade — it is the same bits over fewer lines.
///
/// The schedule follows how much is actually up there to lose. 10-16 kHz bias
/// at the old cutoff: -52.6 dB at 48, -37.2 at 64, -22.0 at 96, -11.5 at 128.
/// Below 64 the sacrifice is of something already absent and the gain is large;
/// by 128 it would be real content given up for a 0.4-point change. Uncapped
/// from 112 up, where the scan's ATH test is the right and only test.
///
/// **The escape hatch is what makes a hard cap safe.** A cap cannot tell
/// residual HF from a signal that lives up there, and on `sweep` — whose whole
/// content is above the cap for part of its duration — the first version took
/// segSNR_p05 from 12.63 to 0.00 at 48 kbps and hf_balance to -96 dB at 32.
/// So lift the cap whenever the region above it carries a real share of the
/// frame's energy: residue never does (that is what makes it residue), and a
/// signal that is genuinely up there always does.
/// Share of the frame's energy that must sit above the cap for it to be lifted.
///
/// Swept on the harness at 32/48/64/96/128. 0.50 is where the fidelity metrics
/// come out clean: segSNR 16 better / 0 worse (mean +0.28), segSNR_p05 11 / 0
/// (+0.26), pre_echo 4 / 0 (+2.25), rate unchanged. Lower is *too permissive* —
/// it lifts the cap on the very sources capping helps most, and at 0.02 the
/// pre-echo win and most of the p05 win disappear (0 / 0 and 7 / 1). Half the
/// frame's energy above 10-13 kHz only happens when the signal really is up
/// there, which is the case worth protecting.
constexpr double LPF_CAP_ESCAPE_SHARE = 0.50;

/// Reading the two metrics that move against this, because both are
/// structurally biased against a narrower band and neither is evidence of harm:
///
///  * `hole_rate_pct` bands the spectrum into 22 Bark bands, so one empty band
///    is 1/22 = **4.55 %** — and that is the exact figure several regressions
///    land on (attack_from_silence@64 0.00 -> 4.55, broadband_noise@64
///    0.04 -> 4.55, @48 0.07 -> 4.55). It is counting the cut, not a hole in
///    coded content. The one large move is the other way and is real:
///    stereo_decorrelated@48 46.81 -> 6.24.
///  * `hf_balance_db` compares energy *above 8 kHz* only, so it cannot see what
///    this buys. Verified directly on mixed_music@32, per band against source:
///    4-6 kHz went from 49 dB down to 6.5 dB down (**+42.6 dB**) and 8-10 kHz
///    from 54 dB down to 11 dB down, while 12-16 kHz — which the old encoder
///    was coding to within 4.4 dB of source at 32 kbps, with the midrange 50 dB
///    down — is now cut. hf_balance reports only the second half of that trade.
///
/// (Unrelated finding from the same measurement, left for later: 6-8 kHz sits
/// 53 dB under the source on mixed_music@32 both with and without the cap.)

double mid_rate_cap_hz(const EncoderState& s) {
  if (std::getenv("RMPS_NO_LPF_CAP")) return 1e9;
  const int br = s.policy_bitr();
  if (br <= 0) return 1e9;              // rate unknown: do not cap
  if (br >= 112) return 1e9;
  double cap;
  if (br >= 96)      cap = 15500.0;
  else if (br >= 80) cap = 14500.0;
  else if (br >= 64) cap = 13500.0;
  else if (br >= 56) cap = 12500.0;
  else if (br >= 40) cap = 12000.0;
  else               cap = 10000.0;

  // Escape hatch: is the signal itself up there?
  const double bin_hz = static_cast<double>(s.wave.samplerate) / 1152.0;
  const int cut_bin = std::min(
      GRANULE_SAMPLES, static_cast<int>(cap / bin_hz + 0.5));
  double e_hi = 0.0, e_tot = 0.0;
  for (int ch = 0; ch < s.wave.channels; ++ch) {
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
      const auto &arr = s.mdct_freq[ch][gr];
      for (int i = 0; i < GRANULE_SAMPLES; ++i) {
        const double v = arr[i];
        const double p = v * v;
        e_tot += p;
        if (i >= cut_bin) e_hi += p;
      }
    }
  }
  static const double escape = [] {
    if (const char *e = std::getenv("RMPS_LPF_ESCAPE")) {
      const double v = std::atof(e);
      if (v >= 0.0 && v <= 1.0) return v;
    }
    return LPF_CAP_ESCAPE_SHARE;
  }();
  if (e_tot > 0.0 && e_hi / e_tot > escape) return 1e9;
  return cap;
}

double lowpass_mid_cutoff_hz(const EncoderState& s) {
  // Experiment hook: RMPS_LPF=<kHz> pins the cut.
  if (const char *e = std::getenv("RMPS_LPF")) {
    const double v = std::atof(e) * 1000.0;
    if (v > 1000.0)
      return v;
  }
  const auto &bt = tables::scale_fact_band_index[s.mpeg.samplerate_index];
  const double sr = static_cast<double>(s.wave.samplerate);
  const double bin_hz = sr / 1152.0;
  const int n_gr = s.mpeg.granules_per_frame;
  const int n_ch = s.wave.channels;
  const double slots = static_cast<double>(std::max(n_gr * n_ch, 1));

  const double floor_hz = lpf_floor_hz(s);
  double cut = floor_hz;
  // Scan to NUM_SFB_LONG *inclusive*. The 21 transmitted scalefactor bands
  // stop at bt[21], which is exactly 16 kHz; the region above it (bt[21]..
  // bt[22], the implicit-sf0 band) is still coded and still audible. Stopping
  // the loop at NUM_SFB_LONG capped the cutoff at 16 kHz and silently removed
  // everything above it at every bitrate.
  for (int sfb = 0; sfb <= NUM_SFB_LONG; ++sfb) {
    const double hi = bt[sfb + 1] * bin_hz;
    if (hi <= floor_hz)
      continue;
    const double lo = bt[sfb] * bin_hz;
    double e = 0.0;
    for (int ch = 0; ch < n_ch; ++ch) {
      for (int gr = 0; gr < n_gr; ++gr) {
        const auto &arr = s.mdct_freq[ch][gr];
        for (int i = bt[sfb]; i < bt[sfb + 1]; ++i) {
          const double v = arr[i];
          e += v * v;
        }
      }
    }
    e /= slots;
    if (psy::power_to_spl(e) >=
        psy::ath_min_db_over(lo, hi) - LPF_ATH_MARGIN_DB)
      cut = hi;
  }
  return std::min(cut, std::min(MID_CUTOFF_HZ, mid_rate_cap_hz(s)));
}

} // namespace

/// Side bandwidth, as a fraction of whatever Mid ended up with.
///
/// Mid comes first. Side carries the stereo difference, and when the budget
/// cannot even hold Mid's own noise floor and tonal detail there is nothing to
/// be gained by spending it on width — a narrower image is a far smaller loss
/// than a grainy centre. So Side's cut tracks Mid's and pulls further down as
/// the rate starves; at comfortable rates it does not bite at all.
double side_cut_fraction(int bitr_kbps) {
  if (bitr_kbps >= 160) return 1.00;   // no extra cut
  if (bitr_kbps >= 128) return 0.85;
  if (bitr_kbps >= 112) return 0.75;
  if (bitr_kbps >=  96) return 0.65;
  if (bitr_kbps >=  80) return 0.55;
  if (bitr_kbps >=  64) return 0.48;
  if (bitr_kbps >=  48) return 0.40;
  if (bitr_kbps >=  32) return 0.32;
  return 0.28;
}

/// Never take Side below this — the image would collapse to mono in a way that
/// is more obvious than the bits are worth.
constexpr double SIDE_FLOOR_HZ = 4000.0;

double lowpass_side_cutoff_hz(int bitr_kbps) {
  // Reported by the CLI before the frame is known, so it answers for the
  // widest Mid the adaptive cut allows.
  return std::max(SIDE_FLOOR_HZ,
                  MID_CUTOFF_HZ * side_cut_fraction(bitr_kbps));
}

/// Mid cutoff hold: never move the cutoff for a single frame.
///
/// The scan re-decides the cutoff every frame from that frame's own spectrum,
/// so a band sitting near ATH at the top flips between coded and cut.
/// Measured with RMPS_LPF_STATS on Witchqueen at 112/160/192 kbps (where the
/// rate cap is off): the cutoff changed on 24.3 % of frames, between 16-18 kHz
/// and 20 kHz+, and 3.8 % of frames were one-frame excursions - a one-frame
/// dropout (or one-frame flash) of the whole top band. That is the same
/// isolated discontinuity the band hole guard stops inside the quantizer, made
/// upstream where it cannot see it: psy runs after the lowpass, so a cut band
/// simply reads as inactive and no hole metric counts it.
///
/// So only move once the scan has asked for the same direction `hold` frames
/// running, and only as far as every frame in that run agreed. hold = 2 is the
/// least that removes one-frame excursions; RMPS_LPF_HOLD=1 disables.
///
/// Once a cut is applied it dwells `hold` frames before a raise may undo it.
/// Without that, "cuts need `hold` requests, raises need one" *creates* the
/// artefact it exists to prevent: when the scan asks for a lower cutoff for
/// exactly two frames, the cut lands on the second and the raise lands on the
/// third, so a two-frame request becomes a one-frame dip. Measured on
/// Witchqueen at 128 kbps: one-frame cuts 1.2 % of frames with the asymmetric
/// hold alone (1.9 % with no hold). The dwell delays a raise only in the frame
/// right after an applied cut, which a sweep's crossing essentially never
/// coincides with.
///
/// A bypass on transient-flagged frames was tried here and removed. It was
/// added when the hold also held raises, to let an attack's top through at
/// once; that need went away when raises stopped being held (below). What it
/// did instead was let *cuts* through: on real music about 30 % of frames
/// carry a transient flag (302 of 1024 on Witchqueen), so one-frame cuts
/// still ran at 1.6 % of frames against 1.9 % with no hold at all.
///
/// And only a *cut* is ever held; a raise goes through at once. The complaint
/// this answers is black spots - a band missing for one frame - and that is a
/// one-frame downward excursion. A one-frame raise is a flash of extra top,
/// not a hole, and raises are exactly what the cap's escape hatch exists to
/// deliver immediately: holding them cost `sweep` segSNR -4.27 dB at 32 kbps
/// and p05 -0.61 / -0.93 at 48 / 64, every time the tone crossed the cap,
/// and the attack bypass above did not help because a sweep is not a
/// transient. Holding only cuts keeps every one-frame dropout out while the
/// return to a lower cutoff simply lands a frame late - a frame of extra top,
/// never a frame of missing top.
static double hold_mid_cutoff(EncoderState &s, double want) {
  static const int hold = [] {
    if (const char *e = std::getenv("RMPS_LPF_HOLD")) {
      const int v = std::atoi(e);
      if (v >= 1 && v <= 16) return v;
    }
    return 2;
  }();
  double &applied = s.lpf_mid_applied_hz;
  if (s.lpf_mid_cut_age < (1 << 20))
    ++s.lpf_mid_cut_age;
  if (applied < 0.0 || hold <= 1) {
    applied = want;
    s.lpf_mid_pending_dir = 0;
    s.lpf_mid_pending_n = 0;
    return applied;
  }
  const int dir = want > applied ? 1 : (want < applied ? -1 : 0);
  if (dir == 0) {
    s.lpf_mid_pending_dir = 0;
    s.lpf_mid_pending_n = 0;
    return applied;
  }
  if (dir > 0) {   // a raise is not held - see above - except by the dwell
    if (s.lpf_mid_cut_age < hold) {
      s.lpf_mid_pending_dir = 0;
      s.lpf_mid_pending_n = 0;
      return applied;
    }
    applied = want;
    s.lpf_mid_pending_dir = 0;
    s.lpf_mid_pending_n = 0;
    return applied;
  }
  if (dir != s.lpf_mid_pending_dir) {
    s.lpf_mid_pending_dir = dir;
    s.lpf_mid_pending_n = 1;
    s.lpf_mid_pending_hz = want;
  } else {
    ++s.lpf_mid_pending_n;
    s.lpf_mid_pending_hz = dir > 0 ? std::min(s.lpf_mid_pending_hz, want)
                                   : std::max(s.lpf_mid_pending_hz, want);
  }
  if (s.lpf_mid_pending_n >= hold) {
    applied = s.lpf_mid_pending_hz;
    s.lpf_mid_cut_age = 0;
    s.lpf_mid_pending_dir = 0;
    s.lpf_mid_pending_n = 0;
  }
  return applied;
}

void apply_lowpass(EncoderState &s) {
  const double nyq = s.wave.samplerate * 0.5;
  const double m_want = std::min(lowpass_mid_cutoff_hz(s), nyq - 1.0);
  const double m_hz = hold_mid_cutoff(s, m_want);
  // Does the Mid cutoff flicker? (RMPS_LPF_STATS=1)
  //
  // It is re-decided every frame from that frame's own spectrum. A band that
  // sits right at the edge can therefore be coded in one frame and cut in the
  // next — and a *one-frame* excursion is exactly the isolated dropout the
  // hole guard exists to stop, produced upstream of it where it cannot see.
  if (std::getenv("RMPS_LPF_STATS")) {
    static double prev1 = -1.0, prev2 = -1.0;
    static long long n = 0, n_change = 0, n_iso = 0;
    static long long n_iso_down = 0, n_iso_up = 0;
    static long long hist[8] = {};   // <10, 10-12, 12-14, 14-16, 16-18, 18-20, 20+
    ++n;
    if (prev1 >= 0.0 && m_hz != prev1) ++n_change;
    // prev1 was a one-frame excursion: its neighbours agree and it does not.
    if (prev2 >= 0.0 && prev1 != prev2 && m_hz == prev2) {
      ++n_iso;
      if (prev1 < prev2) ++n_iso_down; else ++n_iso_up;
    }
    const int b = m_hz < 10000.0 ? 0 : m_hz < 12000.0 ? 1 : m_hz < 14000.0 ? 2
                : m_hz < 16000.0 ? 3 : m_hz < 18000.0 ? 4 : m_hz < 20000.0 ? 5 : 6;
    ++hist[b];
    prev2 = prev1;
    prev1 = m_hz;
    if ((n % 256) == 0)
      std::fprintf(stderr,
                   "[lpf] frames=%lld change=%.1f%% one-frame=%.1f%% (cut %.1f%% flash %.1f%%) | "
                   "<10k %.0f 10-12 %.0f 12-14 %.0f 14-16 %.0f 16-18 %.0f "
                   "18-20 %.0f 20+ %.0f %%\n",
                   n, 100.0 * n_change / n, 100.0 * n_iso / n,
                   100.0 * n_iso_down / n, 100.0 * n_iso_up / n,
                   100.0 * hist[0] / n, 100.0 * hist[1] / n,
                   100.0 * hist[2] / n, 100.0 * hist[3] / n,
                   100.0 * hist[4] / n, 100.0 * hist[5] / n,
                   100.0 * hist[6] / n);
  }
  // Side is bounded by Mid and pulled below it as the rate starves.
  const double s_hz =
      std::min(std::max(SIDE_FLOOR_HZ, m_hz * side_cut_fraction(s.policy_bitr())),
               std::min(m_hz, nyq - 1.0));
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

} // namespace rmps::post
