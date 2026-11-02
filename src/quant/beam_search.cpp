#include "core/scale.hpp"
#include "quant/beam_search.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include "core/huffman_tables.hpp"
#include "core/tables.hpp"
#include "post/masking.hpp"
#include "post/perceptual.hpp"
#include "quant/hf_rdo.hpp"
#include "quant/quantize.hpp"

#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>

namespace rmps::quant {

namespace {

/// Storage bound for the per-band candidate cache. The *active* number of
/// scalefactor candidates is `EncoderState::cfg_sf_max` (from --effort); this
/// only sizes the array, so raising it costs stack whether or not it is used.
constexpr int SF_MAX_CANDIDATE = 5;
/// Storage bound for the rate axis of the scalefactor DP: cumulative estimated
/// bits are bucketed into `cfg_rate_buckets` slots spanning [0, overflow_cap]
/// and the cheapest path per bucket survives. Finer = closer to the exact
/// knapsack solution.
constexpr int NUM_RATE_BUCKETS = 128;

/// Pareto-optimal `scalefac_compress` values: for each pair of per-band
/// scalefactor caps, the cheapest index that provides it. Ordered by part2
/// cost, which is `11·slen1 + 10·slen2` bits for gr=0.
///
///   idx  slen1/slen2  caps (sfb 0-10 / 11-20)  part2
///     5      1 / 1              1 /  1          21
///     8      2 / 1              3 /  1          32
///     9      2 / 2              3 /  3          42
///    11      3 / 1              7 /  1          43
///    13      3 / 3              7 /  7          63
///
/// Every other index is dominated: it either costs more for the same caps or
/// buys range in one half while giving up range in the other for no saving.
/// Notably **14** (4/2 → caps 15/3, 64 bits), one of the two values this
/// search used to consider, is dominated by 13 — one bit cheaper, and 7
/// instead of 3 in the upper half. Since `sf_max` never exceeds 5, nothing
/// above cap 7 is reachable, so 14's wider low half was range the search
/// could not use.
///
/// **0** (0 / 0 -> caps 0 / 0, 0 bits) belongs at the head of this list and was
/// missing. It is not dominated - it is the only zero-cost option - and it is
/// valid whenever every scalefactor of the granule is 0, which is common on
/// quiet Side granules and at low rates. Without it the cheapest the search
/// could ever choose was 5, so every such granule paid 21 bits for
/// scalefactors that were all zero. That was also what turned the black-spot
/// guard's one-line rescue of an empty granule (a 3-bit count1 quad) into a
/// 24-bit spend against grants of 0-25 bits, and at 128 kbps into a frame that
/// overflowed its slot (see the physical-fit guard in iteration_loop).
constexpr std::array<int, 6> COMPRESS_PARETO = {0, 5, 8, 9, 11, 13};

/// The subset of COMPRESS_PARETO worth searching at a given `sf_max`. Caps are
/// clamped by `sf_max` anyway, so two indices that clamp to the same effective
/// caps are the same search — keep the cheaper one only.
///
/// At the default effort `sf_max` is 1, which collapses the whole set to a
/// single value: 5, at 21 bits of part2 against the 42 the old fixed pair
/// spent. That is 21 bits per granule/channel — 84 per stereo frame, ~2.5 %
/// of a 128 kbps frame — bought for nothing, because with sf ∈ {0,1} the caps
/// of 3 that 42 bits paid for were never reachable.
struct CompressSet {
  std::array<int, 6> v{};
  int n = 0;
  const int *begin() const { return v.data(); }
  const int *end() const { return v.data() + n; }
  int operator[](std::size_t i) const { return v[i]; }
};
/// Whether compress 0 (no scalefactor bits) is offered. Opt-in
/// (RMPS_COMPRESS0=1), measured mixed: it is chosen on ~40 % of beam granules
/// at 48 kbps and moves the harness both ways - mixed_music hf_balance
/// +26 dB and segSNR_p05 +1.6 at 32/48, but attack_from_silence@32 hole_rate
/// 12 -> 30 % and transient_train / two_tone_polyphony worse at 64-192. Not
/// needed for stream validity: the physical-fit guard in iteration_loop
/// covers the overspend it was first looked at for.
inline bool compress0_enabled() {
  static const bool on = [] {
    const char *e = std::getenv("RMPS_COMPRESS0");
    return e && std::atoi(e) != 0;
  }();
  return on;
}

CompressSet compress_candidates(int sf_max) {
  CompressSet out;
  for (int idx : COMPRESS_PARETO) {
    if (idx == 0 && !compress0_enabled())
      continue;
    const int c1 = std::min((1 << tables::slen1_tab[static_cast<std::size_t>(
                                 idx)]) - 1,
                            sf_max);
    const int c2 = std::min((1 << tables::slen2_tab[static_cast<std::size_t>(
                                 idx)]) - 1,
                            sf_max);
    bool dup = false;
    for (int k = 0; k < out.n; ++k) {
      const int p = out.v[static_cast<std::size_t>(k)];
      const int p1 = std::min(
          (1 << tables::slen1_tab[static_cast<std::size_t>(p)]) - 1, sf_max);
      const int p2 = std::min(
          (1 << tables::slen2_tab[static_cast<std::size_t>(p)]) - 1, sf_max);
      if (p1 == c1 && p2 == c2) {
        dup = true;
        break;
      }
    }
    if (!dup)
      out.v[static_cast<std::size_t>(out.n++)] = idx;
  }
  return out;
}
constexpr std::array<int, 2> SCALE_CANDIDATES = {0, 1};
constexpr std::array<int, 2> PREFLAG_CANDIDATES = {0, 1};

constexpr std::array<std::array<int, 4>, 4> SCFSI_PATTERNS = {{
    {{0, 0, 0, 0}},
    {{1, 1, 0, 0}},
    {{0, 0, 1, 1}},
    {{1, 1, 1, 1}},
}};

constexpr std::array<int, 5> SCFSI_BOUNDARIES = {0, 6, 11, 16, 21};
constexpr std::array<int, 22> PRETAB = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        1, 1, 1, 1, 2, 2, 3, 3, 3, 2, 0};

int scfsi_band_of_sfb(int sfb) {
  for (int sb = 0; sb < 4; ++sb) {
    if (SCFSI_BOUNDARIES[sb] <= sfb && sfb < SCFSI_BOUNDARIES[sb + 1])
      return sb;
  }
  return 3;
}

int part2_length_for(int compress, const std::array<int, 4> &scfsi_pat,
                     int gr) {
  const int slen1 = tables::slen1_tab[static_cast<std::size_t>(compress)];
  const int slen2 = tables::slen2_tab[static_cast<std::size_t>(compress)];
  int b = 0;
  if (gr == 0 || !scfsi_pat[0])
    b += 6 * slen1;
  if (gr == 0 || !scfsi_pat[1])
    b += 5 * slen1;
  if (gr == 0 || !scfsi_pat[2])
    b += 5 * slen2;
  if (gr == 0 || !scfsi_pat[3])
    b += 5 * slen2;
  return b;
}

// ---- Stack-friendly fixed-size array wrapper to avoid heap allocation ----
struct BandIx {
  std::array<int, 256> data{};
  int size = 0;

  BandIx() = default;
  BandIx(const BandIx &other) : size(other.size) {
    if (size > 0) {
      std::copy_n(other.data.data(), size, data.data());
    }
  }
  BandIx &operator=(const BandIx &other) {
    if (this != &other) {
      size = other.size;
      if (size > 0) {
        std::copy_n(other.data.data(), size, data.data());
      }
    }
    return *this;
  }
  BandIx(BandIx &&other) noexcept : size(other.size) {
    if (size > 0) {
      std::copy_n(other.data.data(), size, data.data());
    }
  }
  BandIx &operator=(BandIx &&other) noexcept {
    size = other.size;
    if (size > 0) {
      std::copy_n(other.data.data(), size, data.data());
    }
    return *this;
  }

  int &operator[](std::size_t idx) { return data[idx]; }
  const int &operator[](std::size_t idx) const { return data[idx]; }
};

// ---- Per-band shaping quantizer (1st-order error feedback) ----
struct QuantOut {
  bool ok = false;
  BandIx ix;
  int max_val = 0;
  std::int64_t scalei_band = 0;
  double dequant_factor = 0.0;
};

/// Largest reconstruction a noise-shaper promotion may produce, as a multiple
/// of the promoted coefficient's own source magnitude. Set once per granule by
/// `beam_search` — tighter around attacks — and read by `quantize_band`'s
/// per-promotion bound. <= 0 disables the bound. Granule-scoped state rather
/// than a parameter because `quantize_band` has six call sites that all quantize
/// the same granule and none of them should be able to disagree about it.
thread_local double g_nsa_promo_ratio = 2.0;
/// Whether this granule is an attack, a run-up, or rises by the shaper bound's
/// threshold (see `beam_search`). Lets other granule-scoped rules share the
/// same notion of "something is arriving here".
thread_local bool g_granule_rising = false;

QuantOut quantize_band(const std::int32_t *xrabs_band, int n,
                       std::int64_t scalei_global,
                       [[maybe_unused]] double steptab_global, int sf,
                       int scale, int preflag, int sfb,
                       const Int2Idx &int2idx, double alpha,
                       double zero_boost_factor = 1.0,
                       double zero_fill_exp = 1.0,
                       const int *prev_ix_band = nullptr,
                       double hyst_scale = 0.0) {
  QuantOut out;
  const int eff_sf = sf + (preflag ? PRETAB[static_cast<std::size_t>(sfb)] : 0);
  const std::size_t sf_idx = static_cast<std::size_t>((1 + scale) * eff_sf);
  {
    static const bool probe = (std::getenv("RMPS_SF_IDX_PROBE") != nullptr);
    if (probe) {
      static std::size_t max_idx = 0;
      static long long n_oob = 0, n_tot = 0;
      ++n_tot;
      if (sf_idx >= 13) ++n_oob;
      if (sf_idx > max_idx) {
        max_idx = sf_idx;
        std::fprintf(stderr,
                     "[sfidx] new max index=%zu (table has 13) n=%lld oob=%lld\n",
                     max_idx, n_tot, n_oob);
      }
    }
  }
  const double sf_mult = tables::pow2_half_tab[sf_idx];
  const std::int64_t scalei = static_cast<std::int64_t>(
      static_cast<double>(scalei_global) * sf_mult + 0.5);
  if (scalei <= 0) {
    out.ix.size = n;
    std::fill_n(out.ix.data.begin(), n, 0);
    out.ok = true;
    out.max_val = 0;
    out.scalei_band = 0;
    out.dequant_factor = 0.0;
    return out;
  }
  if (scalei > 0x7fffffffLL)
    return out;

  const double dequant_factor = 4294967296.0 / static_cast<double>(scalei);

  out.ix.size = n;
  int max_val = 0;
  double err = 0.0;
  // Which coefficients the shaper pushed *above* their own source magnitude.
  // Only these are eligible for the energy cap's rollback: rounding a
  // coefficient up on its own merits is ordinary quantization, not shaping.
  std::array<bool, 256> shaper_raised{};

  for (int j = 0; j < n; ++j) {
    const std::int32_t x = xrabs_band[j];

    // Detect harmonic peaks (local spectral maxima)
    bool is_peak = false;
    if (j > 0 && j < n - 1) {
      is_peak = (x > xrabs_band[j - 1] && x > xrabs_band[j + 1]);
    } else if (j == 0 && n > 1) {
      is_peak = (x > xrabs_band[1]);
    } else if (j == n - 1 && n > 1) {
      is_peak = (x > xrabs_band[n - 2]);
    }

    // Error feedback. `target_raw` is kept unclamped on purpose: it is what
    // the error update below must be measured against, or the loop develops a
    // systematic upward bias.
    //
    // The bias, which is what made the high end sizzle: this filter is
    // supposed to carry a debt forward — code one coefficient short, code the
    // next one long to repay it, and vice versa. But the target is a
    // *magnitude*, so it is clamped at 0, and the clamp is one-sided. When
    // the feedback says "you over-coded, give energy back" hard enough to
    // drive the target negative, the target becomes 0, the coefficient codes
    // to 0, and the old update `err = x_hat - target_f` then read 0 - 0 = 0:
    // the debt was silently discarded. The opposite direction — "you
    // under-coded, code more" — has no such clamp and was always honoured.
    // Only one sign of the correction survived, so the shaper stopped being a
    // noise shaper and became a boost, up to 1.85x (+5.3 dB) at alpha 0.85.
    // High bands feel it worst because most of their coefficients quantize to
    // zero, which is exactly the large-negative-error case that gets amplified
    // into the next coefficient.
    //
    // Measuring the error against the unclamped target keeps the discarded
    // portion in `err`, so it is repaid instead of forgotten and the
    // correction is symmetric again. The shaper can still rescue low-level
    // content — a coefficient rounded away leaves a debt that promotes a
    // later one — it just cannot manufacture energy any more.
    const double target_raw = (alpha > 0.0)
                                  ? (static_cast<double>(x) - alpha * err)
                                  : static_cast<double>(x);
    double target_f = (target_raw < 0.0) ? 0.0 : target_raw;
    shaper_raised[static_cast<std::size_t>(j)] =
        (alpha > 0.0) && (target_raw > static_cast<double>(x));
    // Clamp the top as well. The error-feedback term moves the target *away*
    // from x when `err` is negative — the previous coefficient was coded
    // short — so with alpha up to 0.9 the target reaches 1.9x a coefficient
    // that may already be near full scale. Converting an out-of-range double
    // to int32 is undefined behaviour in C++, and on x86 it yields
    // INT32_MIN: the quantizer then sees a large *negative* magnitude for the
    // loudest bin in the band. Only the lower clamp was here.
    const std::int32_t target_i = fx::sat32(target_f);

    const std::int64_t prod = static_cast<std::int64_t>(target_i) * scalei +
                              static_cast<std::int64_t>(0x80000000LL);
    // Keep the shifted product in 64-bit. `fx::i32` *wraps*, so a coefficient
    // too large for this step size used to come back negative — an overflow
    // that looked like an ordinary small value.
    std::int64_t ln64 = prod >> 32;

    // --- Zero hysteresis ---
    //
    // The 0/1 decision is a bare round-to-nearest at half a step and carries
    // no memory, so a line sitting near that boundary is re-decided from
    // scratch every granule. On sustained material that is audible as
    // harmonics broken into dashes — the artefact that shows up in a
    // spectrogram as a partial blinking on and off.
    //
    // Measured at 48 kbps on real program material: sustained partials at
    // 3-6 kHz move 3.4 dB frame to frame in the *source* and 10.2 dB in the
    // decode, and the frames where a partial vanishes sit only 1.5 dB below
    // its own running median. At 96 kbps the decode moves 4.1 dB, i.e. it
    // tracks the source, and a partial only drops when the source really
    // dipped (-4.8 dB). So at low rates the encoder is re-deciding, not
    // following the material.
    //
    // The deadband is *symmetric*, and that is a measured choice rather than
    // the obvious one.
    //
    // What the transform does, instrumented on real program material: the
    // *source* MDCT's local maxima move 5.5 dB granule to granule, while an
    // STFT of that same source says its partials move 3.4 dB. The MDCT is not
    // shift-invariant, so a steady sinusoid's line magnitude swings with its
    // phase inside the block. That swing is not itself an error — at 96 kbps
    // the decode moves 4.1 dB, i.e. TDAC cancels it on overlap-add and the
    // partial comes out steady. It becomes an audible dropout only when a
    // trough dips below the quantizer's zero threshold, because the
    // cancellation has then lost its partner and the loss is far larger than
    // the one line's own energy.
    //
    // That argues for favouring the "keep alive" direction, and it is wrong.
    // Measured at 48 kbps, 3-6 kHz dropout: symmetric h = 0.1 gives 30.4 % ->
    // 26.8 % (0.2 -> 25.8 %), keep-only gives 29.3 % and does not improve
    // with h at all. The reason is the rate: at CBR the bits are fixed, so
    // promoting lines without also demoting any just drives the beam to a
    // coarser gain elsewhere and the gain cancels. The symmetric deadband
    // promotes exactly the lines it later refuses to promote, which makes it
    // bit-neutral, and that is why it is the one that pays.
    //
    // So: a line that was alive stays alive down to (0.5 - h) steps, a line
    // that was dead needs (0.5 + h) to switch on. h = 0.1 is a 3.5 dB
    // deadband, wider than the 1.5 dB the source moves on the frames where
    // partials were being dropped. RMPS_ZERO_HYST_OFF overrides the "start"
    // half independently, which is how the asymmetric variants were measured.
    //
    // Bounded on purpose: `ln64` is clamped back into {0, 1}, so no larger
    // coefficient moves and the worst case is one step on lines that were
    // already being rounded away. And it is a change to the quantizer's
    // decision boundary, not a re-pricing of the search's cost function —
    // the shape that has repeatedly worked here, where re-weighted cost terms
    // measure inert.
    if (prev_ix_band && hyst_scale > 0.0 && ln64 <= 1) {
      // Fractions of a quantizer step. `keep` lowers the threshold for a line
      // that was alive; `start` raises it for one that was not.
      static const double keep_h = [] {
        if (const char *e = std::getenv("RMPS_ZERO_HYST")) {
          const double v = std::atof(e);
          if (v >= 0.0 && v < 0.5) return v;
        }
        return 0.1;
      }();
      static const std::int64_t keep_bias =
          static_cast<std::int64_t>(keep_h * 4294967296.0);
      static const std::int64_t start_bias = [] {
        double h = keep_h;               // symmetric unless asked otherwise
        if (const char *e = std::getenv("RMPS_ZERO_HYST_OFF")) {
          const double v = std::atof(e);
          if (v >= 0.0 && v < 0.5) h = v;
        }
        return static_cast<std::int64_t>(h * 4294967296.0);
      }();
      const std::int64_t b = static_cast<std::int64_t>(
          hyst_scale *
          static_cast<double>((prev_ix_band[j] != 0) ? keep_bias
                                                     : -start_bias));
      if (b != 0)
        ln64 = std::clamp<std::int64_t>((prod + b) >> 32, 0, 1);
    }

    if (ln64 < 0 || ln64 >= 262144) {
      out.ok = false;
      return out;
    }
    const int v = int2idx[static_cast<std::size_t>(ln64)];
    if (v > 8191) {
      // Too fine a step for this coefficient: 8191 is the largest value the
      // codebook can express. Reject the *candidate* so the search moves to a
      // coarser gain — which it has, and which the `ln64 >= 262144` guard
      // above already handles the same way.
      //
      // This used to clamp to 8191 and carry on. Clamping is a statement
      // about the sample when the problem is with the candidate, and it is
      // silent: the encoder emits a peak coefficient several percent short
      // and its own metrics never object, because they reconstruct through
      // the same clamped index. On a pure tone the peak bin carries
      // essentially all the energy, so a few percent off it *is* the error —
      // a 1 kHz sine at 320 kbps came back at 20.9 dB SNR where LAME gives
      // 79.1, and the ceiling did not move with bitrate because a higher rate
      // buys a *finer* step, which makes the overflow more likely, not less.
      // `quantize()` in quantize.cpp already rejects this case
      // (`return 16384`); this path did not.
      out.ok = false;
      return out;
    }

    out.ix[static_cast<std::size_t>(j)] = v;
    if (v > max_val)
      max_val = v;

    if (alpha > 0.0) {
      const double x_hat = tables::pow43_tab[v] * dequant_factor;
      // Default OFF while the shaper is under review: it is a real fix for a
      // real asymmetry, but it addresses only the branch where the target is
      // driven negative, and measured mixed (segSNR_p05 2 better / 2 worse,
      // hole rate 3 / 1, hf_balance 0 / 2). It does not remove the added HF
      // energy, which comes from the other branch — see the note above
      // `target_raw`. Enabling it mid-investigation would move the baseline
      // the listening tests are being run against.
      static const bool debt_fix = [] {
        const char *e = std::getenv("RMPS_NSA_DEBT_FIX");
        return e && std::atoi(e) != 0;
      }();
      err = x_hat - (debt_fix ? target_raw : target_f);
    }
  }

  // --- Noise-shaper energy cap ---
  //
  // The shaper may rescue, but it may not make the band louder than it is in
  // the source. That constraint already exists for the zero fill a few lines
  // below ("the fill only spends the band's actual energy deficit ... so the
  // reconstructed band can never come out louder than the source"); the
  // shaper simply never had it, and that asymmetry is the whole bug.
  //
  // Why it is needed even though error feedback looks self-correcting: the
  // repayment unit is a whole quantizer level. A run of HF coefficients that
  // round to zero each leave a debt of their own small magnitude, but the
  // only way to pay any of it back is to promote some later coefficient to
  // |ix| = 1 — whose reconstruction is far larger than the coefficients that
  // generated the debt. Every repayment therefore overshoots, and on a
  // zero-heavy band that happens over and over. Measured at 64 kbps the
  // result was a flat +0.9 to +2.1 dB of added energy from 4 kHz up, present
  // on every band, which is what the sizzle was; switching the shaper off
  // removed all of it and switching the noise fill off removed none.
  //
  // So bound the outcome rather than the mechanism. Roll back the promotions
  // the shaper made, most-inflated first, until the band's reconstructed
  // energy is no greater than the source's. Rescue survives untouched: a band
  // coded *below* the source is never modified here, which is the case the
  // shaper exists for.
  if (alpha > 0.0 && n > 0) {
    static const bool cap_off = [] {
      const char *e = std::getenv("RMPS_NO_NSA_CAP");
      return e && std::atoi(e) != 0;
    }();
    if (!cap_off) {
      // Headroom before the cap engages.
      //
      // Exceeding the source band energy by a hair is not a boost, it is
      // ordinary rounding, and at high rates it is most of what the shaper
      // does — the quantizer is fine enough there that the error feedback
      // behaves like the noise shaper it was designed to be. Capping hard
      // clipped that: mixed_music at 320 kbps lost 4.0 dB of segSNR. A
      // tolerance leaves fine adjustment alone and still catches the whole-
      // level overshoot that starved bands produce, which is the audible
      // case and runs to +2 dB.
      static const double tol = [] {
        if (const char *e = std::getenv("RMPS_NSA_CAP_TOL")) {
          const double v = std::atof(e);
          if (v >= 1.0)
            return v;
        }
        return 1.26;   // +1.0 dB
      }();
      double src_e = 0.0;
      double recon_e = 0.0;
      for (int j = 0; j < n; ++j) {
        const double xs = static_cast<double>(xrabs_band[j]);
        src_e += xs * xs;
        const double xh =
            tables::pow43_tab[out.ix[static_cast<std::size_t>(j)]] *
            dequant_factor;
        recon_e += xh * xh;
      }
      // Bounded: each step removes one level from one coefficient, and a
      // coefficient can only be stepped down as far as zero.
      const double budget_e = src_e * tol;
      const int max_steps = 4 * n;
      for (int step = 0; step < max_steps && recon_e > budget_e; ++step) {
        int pick = -1;
        double worst = 0.0;
        for (int j = 0; j < n; ++j) {
          if (!shaper_raised[static_cast<std::size_t>(j)])
            continue;
          const int v = out.ix[static_cast<std::size_t>(j)];
          if (v <= 0)
            continue;
          const double xh = tables::pow43_tab[v] * dequant_factor;
          const double excess = xh - static_cast<double>(xrabs_band[j]);
          if (excess > worst) {
            worst = excess;
            pick = j;
          }
        }
        if (pick < 0)
          break;
        const auto pj = static_cast<std::size_t>(pick);
        const double xh_old = tables::pow43_tab[out.ix[pj]] * dequant_factor;
        const double xh_new =
            tables::pow43_tab[out.ix[pj] - 1] * dequant_factor;
        const double next_e = recon_e - (xh_old * xh_old - xh_new * xh_new);
        // Stop on the side that lands closer to the source energy: undoing
        // one level too many is its own error.
        if (std::abs(next_e - budget_e) > std::abs(recon_e - budget_e))
          break;
        out.ix[pj] -= 1;
        recon_e = next_e;
      }
      max_val = 0;
      for (int j = 0; j < n; ++j)
        max_val = std::max(max_val, out.ix[static_cast<std::size_t>(j)]);
    }

    // --- Per-promotion bound ---
    //
    // The energy cap above bounds the *band*, and that is not enough. A band
    // whose energy is dominated by one loud coefficient has plenty of headroom
    // left for the shaper to promote a quiet neighbour — a coefficient at a
    // tenth of a step, carried over the threshold by accumulated debt, comes
    // back as a whole level, +20 dB over itself — and the band total barely
    // moves, so the cap never looks at it. What that promotion adds is a line
    // the source did not have, at full strength, and a long-block line is a
    // tone across the whole 26 ms window. In an attack granule that tone
    // reaches back in front of the attack: a click just before the hit.
    //
    // Measured on `avicii` (samples in the 5 ms before an onset that exceed
    // the source's own local peak): switching the shaper off entirely takes
    // them from 82 to 46 at 48 kbps and 40 to 19 at 64 kbps, so roughly half
    // are the shaper's. On `shishamo` it makes no difference, so this is one
    // mechanism among several, not the whole story.
    //
    // So bound each promotion against *its own* coefficient. The shaper keeps
    // its real job — a line sitting just under the threshold still gets
    // rescued, since one level over a near-threshold coefficient is only a few
    // dB — and loses the one it was never meant to have, which is inventing
    // lines out of near-silence. Around attacks (`g_nsa_promo_ratio` is set
    // per granule by `beam_search`) the bound is tighter, because that is
    // where an invented line becomes pre-echo.
    //
    // A promoted coefficient is stepped back down, never below what plain
    // rounding of its own source value gives: this only ever removes what the
    // shaper added.
    if (g_nsa_promo_ratio > 0.0) {
      for (int j = 0; j < n; ++j) {
        const auto jj = static_cast<std::size_t>(j);
        if (!shaper_raised[jj] || out.ix[jj] <= 0)
          continue;
        const double xs = static_cast<double>(xrabs_band[j]);
        const std::int64_t ln_plain =
            (static_cast<std::int64_t>(xrabs_band[j]) * scalei +
             static_cast<std::int64_t>(0x80000000LL)) >> 32;
        const int v_plain =
            (ln_plain >= 0 && ln_plain < 262144)
                ? int2idx[static_cast<std::size_t>(ln_plain)]
                : out.ix[jj];
        const double limit = g_nsa_promo_ratio * xs;
        while (out.ix[jj] > v_plain &&
               tables::pow43_tab[static_cast<std::size_t>(out.ix[jj])] *
                       dequant_factor >
                   limit)
          out.ix[jj] -= 1;
      }
      max_val = 0;
      for (int j = 0; j < n; ++j)
        max_val = std::max(max_val, out.ix[static_cast<std::size_t>(j)]);
    }
  }

  // --- Transparent zero fill ("black-spot boost", --hf-zero-boost) ---
  // A coefficient just below the dead-zone boundary quantizes to 0; runs
  // of them decode as spectral holes (黒点). Promote such zeros to |ix|=1
  // with a *level-dependent, compressive* (companding) law — a uniform
  // gain would lift a near-boundary coefficient and a deeply quiet one by
  // the same amount, which over-fills the quiet ones. The quantizer can
  // only write 0 or 1 below the first level, so the curve is realised in
  // fill *density* (expected energy) via error diffusion:
  //   • per-bin fill-energy aim  T = e1 · u · min(1, u·K)  with
  //     u = x/x̂₁ (depth below the level-1 amplitude) and K the energy-
  //     domain boost cap. Near the boundary (u→1) T→e1: the bin is
  //     simply kept, ≈0 dB boost. Deeper bins follow T = e1·u — a lift
  //     of half the depth in dB (compressive: quiet bins gain relatively
  //     more, loud bins less) — until the cap clamps the lift at
  //     +zb dB (T = x²·K), so near-silence is never fabricated into
  //     content;
  //   • an accumulator promotes a bin (|ix|=1) each time the summed aim
  //     reaches e1, loudest first, so the aggregate spectrum follows the
  //     curve exactly even though single bins are all-or-nothing;
  //   • energy cap — the fill only spends the band's actual energy
  //     deficit (source energy minus the coded reconstruction). A band
  //     whose coded bins already sit at the source level gets no fill at
  //     all, so the reconstructed band can never come out louder than
  //     the source.
  if (zero_boost_factor > 1.0) {
    double src_e = 0.0;
    double recon_e = 0.0;
    std::array<int, 256> cand{};
    int n_cand = 0;
    for (int j = 0; j < n; ++j) {
      const double x = static_cast<double>(xrabs_band[j]);
      src_e += x * x;
      const int v = out.ix[static_cast<std::size_t>(j)];
      if (v != 0) {
        const double x_hat = tables::pow43_tab[v] * dequant_factor;
        recon_e += x_hat * x_hat;
        continue;
      }
      if (x > 0.0)
        cand[n_cand++] = j;
    }
    const double budget = src_e - recon_e;
    if (n_cand > 0 && budget > 0.0) {
      std::sort(cand.begin(), cand.begin() + n_cand,
                [&](int a, int b) { return xrabs_band[a] > xrabs_band[b]; });
      const double x1 = tables::pow43_tab[1] * dequant_factor;
      const double e1 = x1 * x1;
      const double K = zero_boost_factor * zero_boost_factor;
      double filled = 0.0;
      double acc = 0.0;
      for (int k = 0; k < n_cand; ++k) {
        const double x =
            static_cast<double>(xrabs_band[cand[static_cast<std::size_t>(k)]]);
        const double u = std::min(1.0, x / x1);
        // zero_fill_exp < 1 flattens the compressive curve (starved
        // bitrates): deeper coefficients then contribute enough aim
        // energy to actually get promoted.
        const double u_curve =
            (zero_fill_exp < 1.0) ? std::pow(u, zero_fill_exp) : u;
        acc += e1 * u_curve * std::min(1.0, u * K);
        if (acc < e1)
          continue;
        if (filled + e1 > budget)
          break;
        out.ix[static_cast<std::size_t>(cand[static_cast<std::size_t>(k)])] = 1;
        filled += e1;
        acc -= e1;
      }
      if (filled > 0.0 && max_val < 1)
        max_val = 1;
    }
  }

  out.ok = true;
  out.max_val = max_val;
  out.scalei_band = scalei;
  out.dequant_factor = dequant_factor;
  return out;
}

double band_distortion(const std::int32_t *xr_band, const BandIx &ix,
                       double dequant_factor, double *out_e_recon = nullptr,
                       double *out_density = nullptr,
                       double *out_deficit = nullptr,
                       double *out_excess = nullptr,
                       double *out_sfm_db = nullptr) {
  double d = 0.0;
  double e_recon = 0.0;
  int nonzero = 0;
  // Per-bin split of the error into the two failures that have opposite
  // cures: source energy that did not survive, and energy that was never
  // there. Done per *bin* rather than per band on purpose — a band that
  // moves energy from one line to another has the right total and is still
  // wrong, and only the per-bin split sees it (the emptied line scores a
  // deficit, the invented one an excess).
  double deficit = 0.0;
  double excess = 0.0;
  for (int j = 0; j < ix.size; ++j) {
    const int v = ix[j];
    if (v != 0)
      ++nonzero;
    double x_hat_mag = tables::pow43_tab[v] * dequant_factor;
    e_recon += x_hat_mag * x_hat_mag;
    double x_hat = x_hat_mag;
    const std::int32_t xr = xr_band[j];
    if (xr < 0)
      x_hat = -x_hat;
    const double diff = static_cast<double>(xr) - x_hat;
    const double sq = diff * diff;
    d += sq;
    // Split `d` itself, rather than comparing band energies. An energy-domain
    // difference (x^2 - x_hat^2) is ~2x(x - x_hat) for a small error — an
    // order of magnitude larger than the squared error for exactly the bins
    // that are nearly right, and proportional to the band's level. Using it
    // as the distortion measure therefore over-weights loud bands and starves
    // quiet ones, which measured as 2.8 dB worse weighted fullness. The sign
    // of the magnitude error is all that is needed to separate the two
    // failures, and splitting `diff*diff` keeps the measure on exactly the
    // same scale as the D_b it replaces.
    if (x_hat_mag < std::abs(static_cast<double>(xr)))
      deficit += sq;      // coded short: source energy that did not survive
    else
      excess += sq;       // coded long: energy that was never there
  }
  if (out_deficit)
    *out_deficit = deficit;
  if (out_excess)
    *out_excess = excess;
  if (out_e_recon)
    *out_e_recon = e_recon;
  if (out_density)
    *out_density =
        (ix.size > 0) ? static_cast<double>(nonzero) / ix.size : 0.0;
  // Fine-structure ratio: spectral flatness of the reconstruction against the
  // flatness of the source over the same bins, in dB. 0 dB = the coded band
  // has the source's texture; strongly negative = the band's energy has been
  // collapsed into a few isolated lines, which is what noise-like HF sounds
  // like when it comes back metallic. See `nf_structure_weight`.
  //
  // Zeros need a floor or one empty bin makes the geometric mean zero. The
  // floor is shared by both spectra and tied to the *source's* mean power, so
  // an emptied line counts as -30 dB in both, and an all-empty band is not
  // mistaken for a flat one.
  if (out_sfm_db) {
    if (ix.size <= 0 || e_recon <= 0.0) {
      *out_sfm_db = -60.0;
    } else {
      double ps_sum = 0.0;
      for (int j = 0; j < ix.size; ++j) {
        const double x = static_cast<double>(xr_band[j]);
        ps_sum += x * x;
      }
      const double ps_mean = ps_sum / ix.size;
      const double pr_mean = e_recon / ix.size;
      if (ps_mean <= 0.0) {
        *out_sfm_db = 0.0;
      } else {
        const double eps = 1e-3 * ps_mean;
        double ls = 0.0, lr = 0.0;
        for (int j = 0; j < ix.size; ++j) {
          const double x = static_cast<double>(xr_band[j]);
          const double xh = tables::pow43_tab[ix[j]] * dequant_factor;
          ls += std::log(x * x + eps);
          lr += std::log(xh * xh + eps);
        }
        const double sfm_s = std::exp(ls / ix.size) / (ps_mean + eps);
        const double sfm_r = std::exp(lr / ix.size) / (pr_mean + eps);
        *out_sfm_db = 10.0 * std::log10(std::max(sfm_r, 1e-12) /
                                        std::max(sfm_s, 1e-12));
      }
    }
  }
  return d;
}

/// Per-band Huffman bit *estimate* — the rate term the RD cost actually uses.
///
/// This replaces a lower bound that charged ~1 bit per non-zero pair: between
/// a third and a tenth of the real Huffman cost, with the shortfall growing
/// with band density. That number fed both the λ_R·bits term of the cost and
/// the budget prune, so the search priced dense (low-frequency) bands as if
/// they were nearly free and only discovered the overspend at the final real
/// bit count — where the whole candidate set can fail at once. Estimate each
/// band as if it were its own Huffman region instead: the best big-values table
/// for the band, and — when |ix| ≤ 1 — the count1 quadruple cost it would get
/// in the granule tail, whichever is cheaper. (The true cost is one of the two,
/// depending on where the big_values boundary lands, which is not knowable
/// per band.) An all-zero band costs 0: it is either the rzero tail, or its
/// cost does not depend on anything the search can choose.
///
/// Two numbers come out of this, because the budget and the trade-off need
/// different ones:
///   `total`    — what the band actually adds to the granule. This is the
///                resource axis of the DP and what the budget is checked
///                against; measured within ~5% of the real coded size.
///   `marginal` — `total` minus what the same band costs when every pair is
///                zero. This is the RD price of *what the search chose*, and
///                it is the one the λ_R·bits term uses. Charging `total`
///                there makes a sparse band (one tone, the rest of the band
///                zero) pay for its zero pairs as if they were content, which
///                over-prices exactly the tonal bands the perceptual model is
///                trying to protect — and those pairs are paid for whether or
///                not the search codes the tone.
struct BandRate {
  int total = 0;
  int marginal = 0;
};

BandRate band_rate(int *b, int n) {
  static constexpr std::array<int, 16> POPCOUNT_16 = {0, 1, 1, 2, 1, 2, 2, 3,
                                                      1, 2, 2, 3, 2, 3, 3, 4};
  BandRate r;
  const int mv = ix_max(b, 0, n);
  if (mv == 0)
    return r;
  const int t = new_choose_table(b, 0, n);
  r.total = count_bit(b, 0, n, t);
  r.marginal = r.total - (n / 2) * huffman::at(t).hlen[0];
  if (mv <= 1) {
    const auto *h32 = huffman::at(32).hlen;
    const auto *h33 = huffman::at(33).hlen;
    int s0 = 0, s1 = 0, i = 0;
    for (; i + 4 <= n; i += 4) {
      const int p = b[i] + (b[i + 1] << 1) + (b[i + 2] << 2) + (b[i + 3] << 3);
      const int sgn = POPCOUNT_16[static_cast<std::size_t>(p)];
      s0 += sgn + h32[p];
      s1 += sgn + h33[p];
    }
    const int n_quad = i / 4;
    const int c1 = (s0 < s1) ? s0 : s1;
    const int base = n_quad * ((s0 < s1) ? h32[0] : h33[0]) +
                     ((n - i) / 2) * huffman::at(1).hlen[0];
    const int tot = c1 + count_bit(b, i, n, 1);
    if (tot < r.total) {
      r.total = tot;
      r.marginal = tot - base;
    }
  }
  if (r.marginal < 0)
    r.marginal = 0;
  return r;
}

BandRate band_rate(BandIx &ix) { return band_rate(ix.data.data(), ix.size); }

// Lightweight stand-in passed to bit-count helpers (mirrors loop.py's
// GrInfoSnapshot).
struct TempGr {
  int big_values = 0;
  int count1 = 0;
  int count1table_select = 0;
  int address1 = 0;
  int address2 = 0;
  int address3 = 0;
  int region0_count = 0;
  int region1_count = 0;
  std::array<int, 3> table_select{0, 0, 0};
};

// We need to call the existing quantize helpers (calc_runlen, count1_bitcount,
// subdivide, bigv_tab_select, bigv_bitcount) which take a `GrInfo&`. Reuse
// them with a real GrInfo we throw away.
int final_bits(int *ix_full, GrInfo &tmp, EncoderState &s) {
  calc_runlen(ix_full, tmp);
  int bits = count1_bitcount(ix_full, tmp);
  // RD-optimal region split (prefix-sum based, cheaper than the legacy
  // fixed-table + heuristic table-select chain). Candidates that only fit
  // the budget under the optimal split become feasible, so the beam can
  // accept finer quantization at the same rate.
  bits += best_subdivide_rd(ix_full, tmp, s);
  return bits;
}

struct CacheEntry {
  bool ok = false;
  BandIx ix;
  double dist = 0.0;
  double D_raw = 0.0;
  double deficit = 0.0;   ///< part of D_raw from coding short
  double excess = 0.0;    ///< part of D_raw from coding long
  double E_recon = 0.0;
  int bits_est = 0;  ///< total band rate — the DP's resource axis
  int rate_cost = 0; ///< marginal band rate — the λ_R·bits term (see band_rate)
  double nsa = 0.0;  ///< shaping alpha this entry was quantized with
  bool is_ext = false; ///< true if quantized from synthesized HF band
};

/// Threshold for "noise-like" content in the upper HF SFBs. Below this
/// tonality we switch the cost from phase-matching to RMS-level-matching.
constexpr double TOP_SFB_NOISE_TONALITY_MAX = 0.30;

/// First SFB that participates in noise-fill mode. At 44.1k sfb 13 ≈ 5 kHz.
/// History: started at sfb 20 (13 kHz+ only), then sfb 17 (~7.5 kHz). Both
/// left the mid-upper band (≈5–11 kHz) to be hard-zeroed by the NMR-hinge /
/// tonality-discount / hf_rdo combination, which showed up as scattered
/// spectral holes ("黒点") in the spectrogram — visibly worse than a
/// reference LAME encode at the same rate. Extending noise-fill down to
/// sfb 13 keeps those noise-like bands continuous (gated by tonality < 0.30,
/// so tonal content is still coded faithfully). Measured segSDR improves at
/// every bitrate (64k +0.2, 96k +0.6, 128k +1.0 dB) with the rate unchanged.
/// Band the HF perceptual weight boost reaches full strength at. This is the
/// value the boost was tuned on; it is kept fixed while the fill's own onset
/// moves, because they are separate mechanisms that only shared a constant.
constexpr int TOP_SFB_HF_BOOST_FIRST = NUM_SFB_LONG - 8;   // 13

/// Band `hf_rdo`'s "stop at the first noise-like band" scan starts from. Also
/// 13, and deliberately independent of the fill's onset — see the note at the
/// scan itself.
constexpr int HF_RDO_SCAN_FIRST = NUM_SFB_LONG - 8;        // 13

/// First scalefactor band the noise fill reaches full strength at.
///
/// Was 13 (NUM_SFB_LONG - 8), which with the two-band ramp put the fill's
/// onset at sfb 12 = **2833 Hz** and full strength at sfb 13 = **3445 Hz** at
/// 44.1 kHz. (The old comment here said "starts biting at sfb 11 - 2.3 kHz,
/// full by 4.1 kHz"; that was off by one band at both ends — the zero branch
/// is `sfb <= first - RAMP`, so sfb 11 gets weight 0, not a partial fill.)
///
/// 2.8 kHz is the presence band, and the fill is not gentle there: its level
/// leg scores a band on whether its *energy* matches the source, so it accepts
/// a coarse quantization whose noise merely sits at the right level, and the
/// injection path seeds outright content into bands that quantized to nothing.
/// Reported audibly as noise bleeding across the midrange at 64 kbps and below.
///
/// Now 16, which puts the onset at sfb 15 = **5130 Hz** and full strength at
/// sfb 16 = **6202 Hz** — 1.8x the old frequencies. The band grid cannot land
/// on exactly 2x: the two candidates bracketing it are 16 (5130/6202, 1.81x)
/// and 17 (6202/7503, 2.19x), equidistant from a doubled 5666/6890.
/// `RMPS_NF_BASE` selects between them.
inline int top_sfb_noise_first() {
  static const int v = [] {
    if (const char *e = std::getenv("RMPS_NF_BASE")) {
      const int x = std::atoi(e);
      if (x >= 2 && x < NUM_SFB_LONG) return x;
    }
    return NUM_SFB_LONG - 5;   // 16
  }();
  return v;
}

/// Smooth noise-fill weight in [0,1] replacing the old binary
/// `sfb >= FIRST && tonality < 0.30` gate. Two soft ramps:
///   • sfb onset — 0 below (FIRST-RAMP), linearly up to 1 at FIRST.
///   • tonality  — 1 below T_LO (very noise-like), down to 0 at T_HI (tonal).
/// The binary gate flipped a band between faithful coding and noise-fill
/// frame-to-frame whenever its tonality dithered across 0.30; that flip-flop
/// is itself a cause of the spectral-hole flicker ("黒点"). A continuous
/// weight makes near-boundary bands settle on a stable partial fill instead.
constexpr double NF_TON_LO = 0.20;
constexpr double NF_TON_HI = 0.40;
constexpr int NF_SFB_RAMP = 2;
/// Lowest scalefactor band the fill may reach, at low rates.
///
/// A rate schedule on top of `top_sfb_noise_first()`, off by default. See that
/// function for where the fill now starts and why. This exists because the
/// midrange bleeding was first fixed with a frequency table before the
/// admissibility conditions (`nf_overshoot_weight`, `nf_density_weight`)
/// replaced it; it is kept as a hook.
constexpr int NF_FIRST_LOW_KBPS  = 64;   ///< floor fully raised at/below
constexpr int NF_FIRST_FULL_KBPS = 96;   ///< normal floor at/above

inline int nf_first_sfb(int policy_kbps) {
  static const int low_first = [] {
    if (const char *e = std::getenv("RMPS_NF_FIRST")) {
      const int v = std::atoi(e);
      if (v >= top_sfb_noise_first() && v < NUM_SFB_LONG)
        return v;
    }
    // Off by default (13 = top_sfb_noise_first() = no schedule). A frequency
    // table was the first fix for the midrange bleeding and it worked, but it
    // is a stand-in: the real question is whether a *particular* fill is
    // acceptable, not whether fills below 4 kHz ever are. The admissibility
    // conditions (nf_overshoot_weight, nf_density_weight) answer that from
    // the candidate itself and measured better than the table at 32/48/64 —
    // hole rate 4 cells better / 2 worse, pre-echo 2 better / 0 worse,
    // segSNR flat — so the table is left in place as a hook and not used.
    //
    // Historical values, if it is ever wanted again:
    // 15 puts the ramp at 3.4-4.9 kHz instead of 2.3-4.1. 17 (~7.3 kHz) is
    // what "high frequencies only" strictly means and is available via the
    // env var, but it costs more at the very bottom of the rate range —
    // measured at 32 kbps, attack_from_silence goes 18.2 -> 40.4 % hole rate
    // at 17 versus 22.5 % at 15, while stereo_decorrelated *gains* (30.5 ->
    // 24.4 % at 48 kbps). 15 is the setting whose regressions stay small.
    //
    // Note that hole rate cannot arbitrate this on its own: filling a hole
    // with substitute noise is precisely how the metric was being satisfied,
    // so removing the fill necessarily makes it read worse while removing the
    // artefact it was hiding. segSNR is flat either way (0 better / 0 worse at
    // 15), so no waveform fidelity is being paid — the fill was substituting,
    // not reconstructing.
    return top_sfb_noise_first();
  }();
  if (policy_kbps <= 0 || low_first <= top_sfb_noise_first())
    return top_sfb_noise_first();
  if (policy_kbps <= NF_FIRST_LOW_KBPS)
    return low_first;
  if (policy_kbps >= NF_FIRST_FULL_KBPS)
    return top_sfb_noise_first();
  const double t = static_cast<double>(policy_kbps - NF_FIRST_LOW_KBPS) /
                   (NF_FIRST_FULL_KBPS - NF_FIRST_LOW_KBPS);
  return static_cast<int>(
      low_first + (top_sfb_noise_first() - low_first) * t + 0.5);
}

/// How dense the quantized band must stay for the fill to be admissible.
///
/// This is the part of the fill decision the search *can* settle for itself,
/// and it is the part the fixed frequency table was standing in for.
///
/// The fill leg scores a band on whether its reconstructed *energy* matches
/// the source, not on the error — so it is happy to accept a coarse
/// quantization whose noise merely lands at the right level. That is a fair
/// description of the result only while enough spectral lines survive for the
/// substitute to actually read as noise. A band of 18 lines coded to two
/// non-zeros is not "noise at the right level"; it is two tones carrying the
/// band's whole energy, which is what bleeding sounds like. Density collapses
/// exactly as the rate starves, which is why the artefact appeared at low
/// bitrates and why a frequency threshold could only ever approximate it.
///
/// Gating the fill on the candidate's own density puts the decision inside
/// the rate-distortion loop: a sparse candidate is scored by the honest
/// waveform leg instead, so the search must either spend the bits to make the
/// band dense or code it faithfully — and D + λ_R·R picks which.
constexpr double NF_DENS_LO = 0.15;   ///< below this the fill is refused
constexpr double NF_DENS_HI = 0.40;   ///< at/above this it is fully admitted

inline double nf_density_weight(double density) {
  static const double lo = [] {
    if (const char *e = std::getenv("RMPS_NF_DENS")) {
      const double v = std::atof(e);
      if (v >= 0.0 && v < 1.0)
        return v;
    }
    return NF_DENS_LO;
  }();
  if (lo <= 0.0)
    return 1.0;                       // gate disabled
  const double hi = std::max(NF_DENS_HI, lo + 1e-6);
  if (density >= hi)
    return 1.0;
  if (density <= lo)
    return 0.0;
  return (density - lo) / (hi - lo);
}

/// The fill's fine-structure admissibility: density replaced by `sfm_ratio`.
///
/// Density — the share of non-zero lines — was standing in for "does this
/// candidate still read as noise". It cannot see how the energy is spread
/// among the survivors, and it knows nothing about the source's own texture.
/// Measured on real material (STFT flatness of the decode against the source,
/// per band, avicii): at 48 kbps 5-8 kHz came back 13.4 dB *peakier* than the
/// source, 8-11 kHz 13.7 dB, 11-14 kHz 47 dB — noise returned as a handful of
/// isolated lines — and switching the fill off moved those by 0.3 dB, i.e. the
/// fill was not addressing it at all.
///
/// So score the candidate on exactly that: its reconstruction's flatness
/// against the source's, over the band's own bins (`band_distortion`). A
/// candidate that keeps the source's texture is admitted to the energy-matching
/// leg; one that collapses it into lines is scored by the waveform leg. The
/// ramp is the lever, not the gate: flatter candidates get more of the cheaper
/// fill scoring, so the search is paid for keeping texture rather than merely
/// told whether it did.
///
/// **Measured inert; default off (density gate).** On avicii and witchqueen at
/// 48 / 64 / 96 kbps every band's sfm_ratio moved by <= 0.4 dB, and the harness
/// was 0 better / 0 worse on every metric — the same result as the density
/// gate itself and as switching the fill off. Deciding which leg scores a band
/// is a blend re-weighting, and those keep measuring inert in this cost. The
/// additive term below the waveform leg was the follow-up.
///
/// RMPS_NF_SFM=1 enables it. RMPS_NF_SFM_LO / _HI move the ramp (dB; refused at
/// or below LO, fully admitted at or above HI).
inline double nf_structure_weight(double density, double sfm_db) {
  static const bool on = [] {
    const char *e = std::getenv("RMPS_NF_SFM");
    return e && std::atoi(e) != 0;
  }();
  if (!on || sfm_db > 1e8)
    return nf_density_weight(density);
  static const double lo = [] {
    if (const char *e = std::getenv("RMPS_NF_SFM_LO")) return std::atof(e);
    return -12.0;
  }();
  static const double hi = [] {
    if (const char *e = std::getenv("RMPS_NF_SFM_HI")) return std::atof(e);
    return -3.0;
  }();
  if (sfm_db >= hi)
    return 1.0;
  if (sfm_db <= lo)
    return 0.0;
  return (sfm_db - lo) / std::max(hi - lo, 1e-6);
}

/// Fill admissibility, the decisive condition: it may not brighten the band.
///
/// The fill's audible defect is that it *adds* high-frequency energy and the
/// result sizzles. That follows directly from how the leg is scored: the cost
/// is `(target_e - E_recon)²`, symmetric, so landing above the source is
/// charged exactly like landing below it — and the injection path, which
/// seeds content into bands that quantized to nothing, has every opportunity
/// to overshoot. Nothing in the model prefers the quiet side of the target.
///
/// Making "does not add energy" an admissibility condition rather than a
/// penalty is what puts this in the search's hands: a candidate that would
/// brighten the band simply loses access to the energy-matching leg and is
/// scored by the honest waveform leg instead, where its added energy is
/// error. The search then has to justify the brightness on D + λ_R·R like
/// anything else, instead of getting it for free.
///
/// This is one-sided on purpose. Undershoot is left alone: a band coded
/// quieter than the source is the hole problem, which the fill exists to
/// solve and the hole surcharge already prices.
constexpr double NF_OVER_RAMP = 1.6;   ///< refused entirely by this ratio

inline double nf_overshoot_weight(double e_recon, double e_band) {
  static const double max_over = [] {
    if (const char *e = std::getenv("RMPS_NF_OVER")) {
      const double v = std::atof(e);
      if (v >= 0.0)
        return v;
    }
    return 1.0;                       // may reach the source level, not exceed
  }();
  if (max_over <= 0.0)
    return 1.0;                       // gate disabled
  const double over = e_recon / std::max(e_band, 1.0);
  if (over <= max_over)
    return 1.0;
  const double hi = max_over * NF_OVER_RAMP;
  if (over >= hi)
    return 0.0;
  return (hi - over) / (hi - max_over);
}

/// Master off-switch for the whole noise-fill mechanism (RMPS_NO_FILL=1).
///
/// Every fill path is gated on this weight — the energy-matching cost leg,
/// the hole injection that seeds content into bands that quantized to
/// nothing, and the zero-boost dead-zone relief — so returning 0 here turns
/// all three off together and the encoder codes purely for waveform error.
/// Provided for A/B: the fill and the noise shaper both add content that was
/// not in the source, and they are easy to mistake for each other by ear.
inline bool fill_disabled() {
  static const bool off = [] {
    const char *e = std::getenv("RMPS_NO_FILL");
    return e && std::atoi(e) != 0;
  }();
  return off;
}

inline double noise_fill_weight(int sfb, double tonality,
                                int first_sfb = -1) {
  if (fill_disabled())
    return 0.0;
  if (first_sfb < 0)
    first_sfb = top_sfb_noise_first();
  double ws;
  if (sfb >= first_sfb)
    ws = 1.0;
  else if (sfb <= first_sfb - NF_SFB_RAMP)
    ws = 0.0;
  else
    ws = static_cast<double>(sfb - (first_sfb - NF_SFB_RAMP)) /
         static_cast<double>(NF_SFB_RAMP);
  double wt = (NF_TON_HI - tonality) / (NF_TON_HI - NF_TON_LO);
  wt = std::clamp(wt, 0.0, 1.0);
  return ws * wt;
}

/// Compressive loudness of a noise-to-mask ratio: N(x) = (1+x)^P − 1.
///
/// P = 0.25 is the Stevens/Zwicker compression exponent — the same power law
/// PEAQ and Moore's partial-loudness models use to turn an excitation ratio
/// into a perceived magnitude. Two properties are why the cost is built on it:
///
///   • x ≪ 1 (noise well under the mask): N ≈ P·x. Small, but *not zero*. The
///     old cost used a hard hinge, `max(0, log(D/Thr))`, which is identically
///     zero below the mask — a flat plateau over every sub-threshold encoding.
///     Any solver that actually reaches that plateau (the DP does; the beam it
///     replaced rarely got there) then picks among them arbitrarily and
///     exploits every error in the threshold estimate to the last dB. That is
///     what cost the tonal clips ~2.5 dB of segSNR. A shallow gradient keeps
///     "inaudible but more accurate" preferred over "inaudible and sloppy"
///     without spending meaningful bits on it.
///   • x ≫ 1: N ≈ x^P grows slowly, so one badly-coded band cannot swamp
///     twenty mildly-noisy ones the way the old linear `D·w` term did.
///
/// The result is dimensionless and mask-relative, so every band — and every
/// granule, and the λ_R·bits term — is finally measured in one unit.
constexpr double LOUDNESS_EXP = 0.25;
inline double noise_loudness(double x) {
  return std::pow(1.0 + std::max(x, 0.0), LOUDNESS_EXP) - 1.0;
}

/// Extra audibility weight for bands below 200 Hz. Replaces a `return D·1e15`
/// branch that removed those bands from the model altogether — masking
/// threshold, bit cost and all — and bought them ~61 dB of band SNR
/// unconditionally. Inside the model they still need a heavy thumb on the
/// scale: the A-weight is small down there and the compressive loudness law
/// flattens differences, so a modest multiplier hands the bass back several
/// dB that measurably do not go anywhere useful. This value is the one the
/// eval harness picked (see the sweep in the session notes); the LF strictness
/// (`--lf-strict`, 18 dB by default) lowers the threshold on top of it.
constexpr double LF_STRICT_WEIGHT = 32.0;

/// Extra weight for granules that contain (or precede) an attack. The old cost
/// gave transient granules a de facto boost by exempting them from a
/// tonality discount that every other granule paid; that discount is gone
/// now — it was double-counting the threshold's own tone/noise offset — so the
/// protection has to be stated directly. Noise inside an attack is exposed
/// both by the lack of a steady masker and by the pre-echo it feeds.
constexpr double TRANSIENT_WEIGHT = 2.0;

/// Weight on the attack granule itself, as opposed to its run-up.
///
/// TRANSIENT_WEIGHT has always applied to both, and for the run-up it is
/// pre-echo protection. On the attack granule it does something else: at low
/// rates, where lambda_R is live, doubling every band's weight buys that one
/// granule a finer quantization than its neighbours, and since HF is what a
/// starved granule gives up first, the extra bits show up as HF. Kept separate
/// so it can be measured on its own. RMPS_TRANSIENT_WEIGHT; default unchanged.
inline double attack_granule_weight() {
  static const double v = [] {
    if (const char *e = std::getenv("RMPS_TRANSIENT_WEIGHT")) {
      const double x = std::atof(e);
      if (x > 0.0 && x < 100.0) return x;
    }
    return TRANSIENT_WEIGHT;
  }();
  return v;
}

/// Exponent of the per-band prominence weight (e_band / e_ref)^q on the
/// waveform/NMR leg. q = 1 reproduces the old cost's `NMR · e_band` factor —
/// "protect whatever is loudest" — and q = 0 drops energy weighting entirely,
/// leaving annoyance to depend on the noise's own level against the mask.
///
/// **This constant is the waveform-SNR ⇄ spectral-hole dial.** Measured over
/// the whole testset (`tools/psyeval.py`), against the same encoder with the
/// rest of this model in place:
///
///   q = 0.0  hole_rate 12 clips better / 3 worse (hf_hiss_decay@64 35.9 % →
///            0.6 %, transient_train@64 38.7 % → 17.1 %), pre-echo 2/0 better,
///            segSNR mean −0.33 dB — almost all of it one cell, mixed_music
///            @192, where the bass drops from 61 to 43 dB band SNR because the
///            bits go to filling holes instead. 43 dB of bass at 192 kbps is
///            far under audibility; 36 % of Bark cells reading 20 dB down is
///            not, which is why this is the default.
///   q = 0.5  mixed_music@192 back to baseline, segSNR mean −0.16 dB, and the
///            hole win gone (mean +0.01). Pick this if waveform SNR is the
///            number that has to hold.
///
/// Nothing else moved the trade: LF weights of 8/32/128, a linear LF cost, the
/// hole surcharge, the loudness exponent and α were all swept and left
/// mixed_music@192 within a dB of −12.
constexpr double PROMINENCE_EXP = 0.0;

/// Surcharge for a spectral hole (a band with energy coded to nothing), as a
/// *multiple of that band's own noise loudness* rather than a flat amount.
/// A gap reads as roughness/flutter beyond what its raw noise power predicts,
/// so it deserves more than the plain NMR — but a flat charge prices a hole in
/// an inaudible band the same as one in a prominent band, and at high bitrates
/// (where every real band's loudness is ~0) that flat term becomes the only
/// thing the search can see: it then buys hole avoidance with precision it
/// should have kept. Proportional pricing keeps holes expensive exactly where
/// they are audible. Noise-like bands scale up to 4× — they lose the most.
/// Base surcharge, in multiples of the band's own noise loudness. Scaled up
/// as the rate starves (see `eff_hole_surcharge`): at 96 kbps and above a
/// heavier charge measurably backfires — the search buys hole avoidance with a
/// coarser global gain and kills cells elsewhere (transient_train at 96 kbps
/// went 3.6 % -> 4.4 % when this was raised from 3 to 10) — while below that
/// the starved encodes want all of it.
constexpr double HOLE_SURCHARGE = 3.0;

/// Asymmetric over-shoot charge for the high bands.
///
/// Every other leg of this cost is *symmetric* in the sign of the level error:
/// the waveform leg squares the per-bin error, and the noise-fill leg squares
/// `target_e − E_recon`. Coding a band 3 dB hotter than the source therefore
/// costs exactly what coding it 3 dB quieter costs. Perceptually it does not.
/// An HF band left quiet reads as a slightly dull top end; the same band
/// coded loud reads as hiss/sizzle that was never in the source, and it is
/// the direction this encoder is structurally biased toward up there:
///
///   • `target_e` (the noise-fill aim point) is allowed up to `min(thr, 2·e)`,
///     i.e. +3 dB of masked headroom that costs nothing;
///   • the noise injector's `can_inject` allowance relaxes from +0.5 dB to
///     ~+2 dB as the rate starves, deliberately preferring a hot fill to a
///     hole;
///   • `hf_boost` / the pre-window ramp raise `weight_eff` in exactly these
///     bands, so the search is pushed hard to *not* leave them under-filled,
///     with nothing pushing back on the far side.
///
/// So charge the over-shoot separately and in one direction only, on top of
/// whatever the symmetric legs already said. The surplus energy
/// `E_recon − e_band` is by definition not in the source, so it is measured
/// as noise against the mask and run through the same loudness law as every
/// other term — it stays in the one unit, and it is identically zero for any
/// band coded at or under its source level.
constexpr int HF_OVER_FIRST_SFB = 11; ///< ~2.6 kHz at 44.1k; same gate hf_rdo
                                      ///< and the pre-window ramp use.
constexpr int HF_OVER_RAMP = 2;       ///< SFBs of fade-in below the gate.
/// Dead-band: quantization to the nearest step routinely lands a band a
/// fraction of a dB hot, and the +3 dB of masked headroom is a deliberate
/// choice, not an artefact. Only charge past this much over the source.
/// Stored as a power ratio (= 10^(1.5/10)) so the hot path does no pow().
///
/// Flat, deliberately. Widening it as the rate starves — to match the noise
/// injector's own hot-fill allowance, which relaxes from +0.5 dB to ~+2 dB
/// with `low_rate_t` — was tried and reverted: every band that actually
/// over-shoots above 8 kHz on the testset does so at 64-96 kbps, so the ramp
/// disarmed the term exactly where it is needed (stereo_correlated@96 went
/// straight back to +1.59 dB) and did not reliably buy back the hole rate it
/// was meant to protect (attack_from_silence@64 got worse, not better).
constexpr double HF_OVER_DEADBAND = 1.4125; ///< +1.5 dB
/// Strength, as a multiple of the band's ordinary perceptual weight. 1.0 would
/// price invented HF energy exactly like coding noise of the same power; the
/// term needs to be *louder* than that to change any decision, because it
/// competes with `hf_boost` and the noise-fill leg pushing the other way.
///
/// Measured against the same encoder without the term (tools/psyeval.py, whole
/// testset):
///
///   0.6  Safe and nearly inert. Every over-shooting cell improves by only
///        0.03-0.05 dB; nothing anywhere regresses past a harness threshold.
///   3.0  The default. Over-shooting cells move 0.06-0.10 dB toward 0
///        (stereo_correlated@64 5.00 → 4.96, @96 1.58 → 1.49, harmonic_tone
///        @64 2.54 → 2.44, broadband_noise@64 0.20 → 0.14), and mixed_music
///        @192 gains 1.1 dB segSNR / 2.3 dB p05 with hf_balance −0.81 → −0.29.
///        The cost is 1.1 points of hole rate on hf_hiss_decay@64 — the
///        starved-HF clip, where refusing a hot fill leaves a gap instead.
///        That is the 高周波オーバー ⇄ 黒点 trade; drop to 0.6 if the hole
///        rate at 64 kbps is the number that has to hold.
///
/// Going further is not a knob, it is a different term: at 3.0 with the
/// dead-band removed the search starts buying quiet HF outright — the sweep
/// clip lost 1.9 dB of hf_balance and 0.4 dB of segSNR at 192 kbps, because
/// a gliding tone lands hot in its leading band every granule and the only
/// way to never be hot is to be dark.
constexpr double HF_OVER_WEIGHT = 3.0;

/// Bit headroom reserved for the implicit-sf0 region above the 21 transmitted
/// scalefactor bands (bins 418-576 = 16.0-22.05 kHz at 44.1k).
///
/// That region is spliced in at the very end of the search, and kept only if
/// it happens to fit in what the 21 coded bands left over. Measured, it
/// almost never does — and the problem gets *worse* as the rate rises,
/// because a richer budget is a budget the DP spends in full:
///
///   broadband_noise  128 kbps  kept 4/229    192: 3/229    320: 0/229
///   hf_hiss_decay    128 kbps  kept 0/229    192: 0/229    320: 0/229
///
/// At 320 kbps every granule of both clips was coded with everything above
/// 16 kHz zeroed, short by 400-650 bits out of ~3000. Nothing below is
/// remotely starved at that rate; the region loses only because it asks last.
///
/// So take its bits off the top instead: measure what the region costs on the
/// seed-gain granule, hold that back from the budget the search is allowed to
/// spend, and let the splice claim it afterwards. The slack factor is there
/// because the reservation is only useful if it is *comfortably* enough — the
/// chosen gain is not the seed gain, and one bit short means the whole region
/// is zeroed exactly as before.
///
/// Deliberately *not* gated on bitrate. The affordability ceiling below is the
/// gate: at a starved rate the region's estimate blows past it and nothing is
/// reserved, at a rich one it fits and the region gets emitted. That is the
/// same question a bitrate threshold was a proxy for, asked against the actual
/// content instead of the rate — a 64 kbps granule whose >16 kHz region is
/// nearly empty can afford it, and a 320 kbps granule of bright noise may not.
constexpr double HF21_RESERVE_SLACK = 1.15;
/// Affordability ceiling, as a fraction of the granule budget. Above this the
/// region is simply not worth what it costs and *no* reservation is made (see
/// the all-or-nothing rule at the estimate). The reservation must never be
/// able to starve the audible bands, whatever the estimate says — and it
/// competes with them globally, not just inside its own granule, because the
/// bits it does not return go through the reservoir to the next one.
///
/// The ceiling moves with the rate, because what counts as affordable does.
/// A third of a 64 kbps granule is not affordability, it is robbery: measured
/// flat at 0.35, attack_from_silence@64 went from 1.9 % to 51.8 % hole rate
/// and stereo_decorrelated@96 from 0.7 % to 8.2 %, because everything audible
/// was paying for 16-22 kHz. 0.08 down there still lets a granule whose top
/// region is genuinely cheap through — which is the whole rule: emit it
/// whenever it can be emitted, at any bitrate, and let the cost decide.
///
/// The high end is 0.25 rather than the 0.35 that the region's own size
/// (bins 418-576 = 27 % of the coefficients) would justify, because 0.35 is
/// where it starts taking bits from content that needs them more:
///
///   0.25  every gain below except hf_hiss_decay, and *no* metric regresses
///         anywhere on the testset (p05 8 better / 0 worse).
///   0.35  additionally serves the brightest material — hf_hiss_decay@192
///         hf_balance −5.97 → −2.31 dB with +2.2 dB segSNR — but its region
///         costs 33-35 % of the granule, and paying that on transient_train's
///         long-block click granules (27-32 %, and their top region really
///         does hold 60 % as much energy as every coded band combined) costs
///         7.5 dB of worst-case segSNR at 192 kbps and 7.3 dB at 320.
///
/// Raise it to 0.35 if bright sustained HF matters more than transient
/// worst-case; nothing else in the model needs to change with it.
constexpr double HF21_RESERVE_MAX_FRAC_LO = 0.08; ///< at/below 96 kbps
constexpr double HF21_RESERVE_MAX_FRAC_HI = 0.25; ///< at/above 192 kbps
constexpr int HF21_FRAC_LO_KBPS = 96;
constexpr int HF21_FRAC_HI_KBPS = 192;

/// Strength of the fidelity reward — the term that prefers a smaller raw
/// difference from the source whenever the perceptual model is indifferent
/// (see its use in `hybrid_band_cost`). Sized to be decisive on ties and
/// invisible otherwise.
///
/// **Why the shape is logarithmic.** The first version used `x/(1+x)`, which
/// is well behaved in the abstract but not against this encoder's numbers.
/// Measured on the chosen path (mixed_music, RMPS_EREF_DEBUG):
///
///     64 kbps   x_abs mean 3.7e-02   max 1.3e-01
///    192 kbps   x_abs mean 1.5e-03   max 6.1e-03
///    320 kbps   x_abs mean 1.4e-04   max 9.2e-04
///
/// Two things follow. x_abs never approaches 1, so `x/(1+x)` is just `x` —
/// the saturation that made it safe never engages, and the term is linear in
/// a quantity of order 1e-3. And x_abs falls ~100× from 64 to 320 kbps,
/// because `e_ref` is the granule's loudest band while the error shrinks with
/// the rate — so *any* fixed multiplier makes the term rate-dependent:
/// dominant where the encode is already coarse, invisible where it is fine.
///
/// Shrinking `e_ref` itself (the obvious fix — "just divide it down") cannot
/// work either. e_ref measures 1e14-1e18 here, so a small constant reference
/// pushes x_abs to ~1e13: the reward saturates to exactly 1 for every band,
/// carrying no information at all, while `waveform_reg` and the α=0 leg —
/// which share x_abs — grow to hundreds of times every other term and switch
/// the masking model off in all but name.
///
/// `log2(1 + x/X0)` was the obvious answer to both and was tried: it never
/// saturates, and being logarithmic it pays the same reward for 6 dB of band
/// SNR whether the band started at 20 dB or 60 dB. Measured, it is worse than
/// what it replaced — 0 cells better, 5 worse on segSNR, 1/4 on p05 — and the
/// reason is that same equal-pay-per-dB property: it pulls bits toward bands
/// that are already transparent, because another 6 dB there scores exactly as
/// well as 6 dB in a band that needs them. "Lower is always better" is true
/// per band and false as an allocation rule.
///
/// So the shape stays `x/(1+x)`, which in this range is effectively linear in
/// x_abs and therefore weights *large* differences more — the allocation the
/// measurement prefers — with the saturation held in reserve for bands that
/// blow up. 4.0 is where it stops being inert: at 0.25 it measured 0 cells
/// better and 0 worse, three orders of magnitude under the costs it is meant
/// to break ties between.
/// Sweepable with RMPS_SRC_DIFF for measurement.
const double SRC_DIFF_REWARD = [] {
  if (const char *e = std::getenv("RMPS_SRC_DIFF")) {
    const double v = std::atof(e);
    if (v >= 0.0) return v;
  }
  return 4.0;
}();

/// λ_R arrives on the legacy scale (masking.cpp: `(128-br)/128 · 1e9`, so 0 at
/// ≥128 kbps and 5e8 at 64 kbps) because it used to be added to costs of order
/// 1e14. The loudness cost is order 1, so bring λ into that unit: at 64 kbps
/// this makes λ_R·bits ≈ 1 for a ~200-bit band, i.e. comparable to a band
/// sitting well over its mask. The term is finally live — it was inert before
/// (scaling it 100× either way produced bit-identical files).
constexpr double LAMBDA_LOUDNESS_SCALE = 1.0e-11;

/// Per-band cost, in one unit: the partial loudness of the coding noise.
///
///     cost = w_b · N( α·D/Thr + (1-α)·D/E ) + holes + λ_R·bits
///
///  - `α` keeps its documented meaning: 1 = pure mask-relative (NMR), 0 =
///    legacy waveform distortion. The α=0 leg normalises by a *granule-level*
///    reference energy, not by the band's own threshold, so inside a granule
///    it stays proportional to D — the same ordering the old `D·w` term had.
///    (Normalising it per band instead makes every band equally worth
///    protecting regardless of level, which measured −11 dB of segSNR on
///    mixed_music at 192 kbps: the surplus bits went to quiet bands.)
///  - The tonality discount that used to multiply this whole expression is
///    **gone**. It was double-counting: `masking.cpp` already lowers a tonal
///    band's threshold by the tone/noise offset (TMN 14.5 dB vs NMT 5.5 dB,
///    `psy::smr_db`) before spreading, so tonal content was protected twice
///    and noise-like content discounted twice. The psychoacoustics belongs in
///    the threshold; the cost's job is only to measure distance from it.
///
/// The **noise-fill** regime (high, noise-like bands) keeps its own leg: there
/// the cost compares reconstructed band *energy* to the source level instead
/// of per-bin waveform error, so the search may pick a coarse quantization
/// that lands noise at the right level — perceptually equivalent to the
/// source's own hiss. Same loudness unit, so the two legs blend smoothly.
inline double hybrid_band_cost(int sfb, double D_b, double E_recon_b,
                               double thr_b, double e_band_b, double tonality_b,
                               double weight_b, int bits_est, double alpha,
                               double lambda_R, bool granule_is_transient,
                               double hf_boost, bool is_pre_window = false,
                               double e_ref = 1.0, double energy_contrast = 1.0,
                               bool is_below_200 = false,
                               double waveform_reg = 0.0,
                               bool side_channel = false, bool is_comb = false,
                               double hole_surcharge = HOLE_SURCHARGE,
                               int nf_first = -1,
                               double nf_density = 1.0,
                               double deficit_e = -1.0,
                               double excess_e = 0.0,
                               double nf_sfm_db = 1e9) {
  // --- Asymmetric error: the "adding" direction costs 25x ---------------
  //
  // `D_b` is `deficit + excess`, computed per bin by `band_distortion`, so
  // every leg below prices a line coded 3 dB short exactly like one coded
  // 3 dB long. They are not alike. A deficit sounds like less of the music; an
  // excess is energy the source never had — a line the noise shaper invented,
  // a promotion that overshot — and when it lands in an attack granule it is a
  // click in front of the hit rather than a slightly duller hit.
  //
  // So rebuild the error with the excess weighted 25x before anything uses it.
  // Doing it here, at the entry, rather than inside one leg is deliberate:
  // re-scaling bleed *inside* the Fullness/Bleedless max() was measured inert
  // at 1.5x, 3x and 6x (see that block), and in this encoder re-weighted
  // blends keep turning out inert while additive legs move allocation. With the
  // rebuilt `D_b` the asymmetry reaches the mask leg, the absolute leg, the
  // additive `waveform_reg` leg and the fidelity reward alike, as well as the
  // bleed leg.
  //
  // The hard bleed barrier further down keeps the *unscaled* excess: it is a
  // limit on an actual mask-relative level, and moving its units would move
  // the limit.
  //
  // RMPS_EXC_MULT sets the factor; 1 is the symmetric error.
  //
  // **Default 1 — measured, not adopted.** At 25x everywhere the harness got
  // worse on every quality metric against the shaper bound alone: segSNR
  // 1 better / 11 worse, segSNR_p05 0 / 13, hole_rate 0 / 13 (hf_hiss_decay
  // -15 points and transient_train -13 at 96 kbps). The search stops risking
  // any excess and rounds down instead, which is the fullness-for-bleedless
  // trade the bleed barrier below already documents. On the clicks it is
  // mixed: shishamo's pre-attack clicks drop (90 -> 80 at 48 kbps, 62 -> 48 at
  // 96), which the shaper bound does not reach, but combined with the bound it
  // *doubles* avicii's (18 -> 34 at 48 kbps). Scoped to rising granules
  // (RMPS_EXC_SCOPE=2) it is cheaper — segSNR_p05 3 / 7, hf_balance 1 / 6 — and
  // keeps most of the shishamo gain (62 -> 42 at 96 kbps).
  static const double exc_mult = [] {
    if (const char *e = std::getenv("RMPS_EXC_MULT")) {
      const double v = std::atof(e);
      if (v > 0.0 && v <= 1000.0) return v;
    }
    return 1.0;
  }();
  // Scope: RMPS_EXC_SCOPE=0 applies the factor everywhere; 1 only on flagged
  // attack and run-up granules; 2 on those plus any granule whose energy rises
  // (the same test the shaper's promotion bound uses).
  static const int exc_scope = [] {
    if (const char *e = std::getenv("RMPS_EXC_SCOPE")) return std::atoi(e);
    return 0;
  }();
  const bool exc_here =
      exc_scope == 0 ||
      (exc_scope == 1 && (granule_is_transient || is_pre_window)) ||
      (exc_scope == 2 && g_granule_rising);
  const double excess_raw = excess_e;
  if (deficit_e >= 0.0 && exc_mult != 1.0 && exc_here) {
    D_b = deficit_e + exc_mult * excess_e;
    excess_e *= exc_mult;
  }

  // --- Level-dependent Fullness / Bleedless weighting --------------------
  //
  // The two directions of error do not cost the same at every level. In a
  // quiet band — a reverb tail, a fade, the space between notes — losing what
  // is there is the failure that is heard: the band drops out and the texture
  // collapses to silence. Adding a little is the lesser failure there. At
  // full level the balance is the usual one. So weight the deficit up and the
  // excess down as the band's own level falls, and leave loud bands alone.
  //
  // Level is the source band's energy in dBFS of the *input*, calibrated with
  // test sines through the whole chain (pre-gain included): a full-scale 1 kHz
  // sine puts 10*log10(e_band) = 183.6 into its band at headroom shift 15, and
  // -20 / -40 / -60 dBFS sines land 20.0 / 40.0 / 60.2 dB below that. The
  // shift-dependent part tracks `scale::mdct_scale_db`, as every other MDCT-
  // domain constant here does.
  //
  // Shape. With u = (HI - level) / (HI - LO) — 0 at HI, 1 at LO, and past 1
  // below LO — the deficit weight is DEF_MAX^(u^GAMMA), capped at DEF_CAP.
  // GAMMA > 1 makes it gentle just under HI and steep toward the floor, and u
  // is not clamped at 1, so the protection keeps growing below LO instead of
  // levelling off exactly where content is most at risk of being cut. At the
  // defaults (GAMMA 2, DEF_MAX 25): -45 dBFS x2.2, -60 x25, -75 x1400, capped
  // at x1e4 by about -78. The excess weight follows the same u but stops at
  // EXC_MIN: relaxing excess without bound is how the attack clicks came back
  // in the linear version's measurement (avicii@48k pre-attack spikes 18 -> 23).
  // GAMMA 1 with DEF_CAP = DEF_MAX reproduces that linear version exactly.
  // Applied at the entry, like the excess multiplier above, so the weights
  // reach the additive legs and not only the Fullness/Bleedless max().
  //
  //   RMPS_FB_LVL=0           disable
  //   RMPS_FB_LVL_HI_DB       -30   (dBFS, above: unchanged)
  //   RMPS_FB_LVL_LO_DB       -60   (dBFS, u = 1)
  //   RMPS_FB_LVL_DEF_MAX     25    (deficit weight at u = 1)
  //   RMPS_FB_LVL_GAMMA       2
  //   RMPS_FB_LVL_DEF_CAP     1e4
  //   RMPS_FB_LVL_EXC_MIN     0.25
  {
    // **Default off — measured.** Harness against off: linear segSNR 1 / 4,
    // segSNR_p05 2 / 5, hole_rate 5 / 5 (better at 48-64 kbps, worse at
    // 128-192); GAMMA 2 segSNR 0 / 5, p05 3 / 6; GAMMA 3 the same and it
    // tripled avicii's pre-attack clicks at 48 kbps (18 -> 58). On the direct
    // measure — share of source cells at -60..-50 dBFS the decode cuts by
    // >= 10 dB, avicii@48k — every shape landed within a point of off (11.5 %
    // off, 12.5 linear, 12.3 GAMMA 2, 12.1 GAMMA 3). The reason is the
    // candidate set, not the price: at the default effort a band's
    // scalefactor is 0 or 1, so a quiet band beside a loud one cannot be made
    // fine enough to survive whatever it costs. Widening that range is what
    // moves it (sf_max 1 -> 3: 11.5 -> 8.2 %, 5: 7.5 %), and adding this
    // weighting on top of the wider range made the cut rate *worse*
    // (-70..-60 dBFS 17.0 -> 21.2 % at sf_max 3). RMPS_FB_LVL=1 enables.
    static const bool lvl_on = [] {
      const char *e = std::getenv("RMPS_FB_LVL");
      return e && std::atoi(e) != 0;
    }();
    auto envd = [](const char *name, double def) {
      if (const char *e = std::getenv(name)) return std::atof(e);
      return def;
    };
    static const double hi_db = envd("RMPS_FB_LVL_HI_DB", -30.0);
    static const double lo_db = envd("RMPS_FB_LVL_LO_DB", -60.0);
    static const double def_max = envd("RMPS_FB_LVL_DEF_MAX", 25.0);
    static const double exc_min = envd("RMPS_FB_LVL_EXC_MIN", 0.25);
    static const double gamma = envd("RMPS_FB_LVL_GAMMA", 2.0);
    static const double def_cap = envd("RMPS_FB_LVL_DEF_CAP", 1.0e4);
    if (lvl_on && deficit_e >= 0.0 && hi_db > lo_db && def_max > 0.0 &&
        exc_min > 0.0) {
      constexpr double FULL_SCALE_BAND_DB_AT_REF = 189.66;  // 183.64 at shift 15
      const double fs_db = FULL_SCALE_BAND_DB_AT_REF + scale::mdct_scale_db();
      const double level_db = 10.0 * std::log10(std::max(e_band_b, 1.0)) - fs_db;
      const double u = std::max(0.0, (hi_db - level_db) / (hi_db - lo_db));
      if (u > 0.0) {
        const double g_def =
            std::min(std::pow(def_max, std::pow(u, gamma)), std::max(def_cap, 1.0));
        const double g_exc = std::pow(exc_min, std::min(u, 1.0));
        deficit_e *= g_def;
        excess_e *= g_exc;
        D_b = deficit_e + excess_e;
      }
    }
  }

  // Continuous noise-fill weight (replaces the old binary
  // `sfb>=FIRST && tonality<0.30` switch). w_nf ≈ 1 for noise-like HF
  // bands, ≈ 0 for tonal / low bands, with smooth ramps on both axes.
  // Pre-window granules: suppress noise-fill entirely. Filling the quiet
  // pre-attack region with noise defeats the purpose of pre-echo control.
  // `nf_first` raises the fill's lowest band as the rate starves (see
  // nf_first_sfb). Scaling the weight down uniformly was tried first and is
  // inert — both legs of the blend are monotone in the error, so re-weighting
  // them leaves the ranking alone. Moving the *boundary* is not inert: it
  // changes which leg a band is scored by at all.
  const double w_nf = (granule_is_transient || is_pre_window || is_comb)
                          ? 0.0
                          : noise_fill_weight(sfb, tonality_b, nf_first) *
                                nf_structure_weight(nf_density, nf_sfm_db) *
                                nf_overshoot_weight(E_recon_b, e_band_b);
  // Apply the HF weight boost to noise-fill SFBs. Multiplies the perceptual
  // axis only — bit cost is untouched so the search can still trade bits for
  // HF preservation. Graded by the sfb ramp so there is no hard step.
  // Note this is *not* the fill's band any more. The two mechanisms shared one
  // constant until the fill's onset was raised; the boost is a perceptual
  // re-weighting of the upper spectrum and has no reason to follow it, so it
  // keeps the band it was tuned on.
  const double boost_w =
      (sfb >= TOP_SFB_HF_BOOST_FIRST) ? 1.0
      : (sfb <= TOP_SFB_HF_BOOST_FIRST - NF_SFB_RAMP)
          ? 0.0
          : static_cast<double>(sfb -
                                (TOP_SFB_HF_BOOST_FIRST - NF_SFB_RAMP)) /
                NF_SFB_RAMP;
  double weight_eff = weight_b * (1.0 + boost_w * (hf_boost - 1.0));

  // Pre-window HF penalty: boost the perceptual weight for HF SFBs so
  // the beam search keeps the HF cleaner in the pre-echo zone. The boost
  // is graded by the energy contrast (sharper attack → more exposed
  // pre-echo → stronger penalty). SFB 11+ gets the boost, ramping up.
  if (is_pre_window && sfb >= 11) {
    // Smooth ramp: sfb 11→16 get 2.0x..3.0x, sfb 17→20 get 3.0x..5.0x.
    double pe_boost = 1.0;
    if (sfb >= 17)
      pe_boost = 3.0 + 2.0 * (sfb - 17) / std::max(20 - 17, 1);
    else
      pe_boost = 2.0 + 1.0 * (sfb - 11) / std::max(16 - 11, 1);
    // Scale by energy contrast (diminishing returns via log).
    if (energy_contrast > 1.0)
      pe_boost *= 1.0 + 0.5 * std::log(energy_contrast);
    weight_eff *= pe_boost;
  }

  // Below 200 Hz the ear's frequency selectivity is poor and the A-weight is
  // low, so the plain model under-protects the bass. Weight it up instead of
  // exempting it from the model (see LF_STRICT_WEIGHT).
  if (is_below_200)
    weight_eff *= LF_STRICT_WEIGHT;
  if (is_pre_window)
    weight_eff *= TRANSIENT_WEIGHT;
  else if (granule_is_transient)
    weight_eff *= attack_granule_weight();

  const double thr = std::max(thr_b, 1.0);

  // Level-matching ("noise-fill") leg. The aim point tracks the source's own
  // band level, with the masked headroom capped at +3 dB: letting a generous
  // threshold pull the fill above the source decodes as audible extra hiss.
  // The level error is charged as an equivalent noise power, so it lands in
  // the same loudness unit as the waveform leg.
  // The headroom multiplier is the one thing that decides how much *invented*
  // energy the fill may add. Blending the fill regime in or out (w_nf) does
  // not: both legs are monotone in the error, so re-weighting them leaves the
  // search's ranking alone — measured inert even at a 67 % reduction of the
  // fill weight at 64 kbps. The aim point is not monotone in that way; it
  // moves where the fill is trying to land.
  static const double nf_ceiling = [] {
    if (const char *e = std::getenv("RMPS_NF_CEIL")) {
      const double v = std::atof(e);
      if (v >= 1.0)
        return v;
    }
    return 2.0;
  }();
  const double target_e =
      std::max(e_band_b, std::min(thr_b, nf_ceiling * e_band_b));
  const double level_diff = target_e - E_recon_b;
  const double x_level =
      (level_diff * level_diff) / std::max(target_e, 1.0) / thr;
  const double p_level = weight_eff * noise_loudness(x_level);

  // Faithful ("phase") leg: noise measured against the mask (α) and against
  // the band's own signal level (1-α). Both are ratios, so both are
  // scale-free; a band quieter than its own threshold falls back to the
  // mask-relative measure via the max() floor.
  const double x_mask = D_b / thr;
  const double x_abs = D_b / std::max(e_ref, 1.0);
  double x_phase = alpha * x_mask + (1.0 - alpha) * x_abs;

  // --- Fullness / Bleedless distortion measure --------------------------
  //
  // Replaces *only* the distortion measure, leaving every weight around it
  // in place — prominence, the hole surcharge, LF strictness, the pre-echo
  // and transient weights, the noise-fill blend. An earlier attempt returned
  // a complete cost from here and bypassed all of that; it measured 2.9 dB
  // worse on fullness, because those weights are accumulated fixes, not
  // decoration.
  //
  // The measure itself: `D_b` sums the squared error and cannot tell the two
  // failure modes apart, but they have opposite cures. `deficit_e` is source
  // energy that did not survive (a hole, a lost octave); `excess_e` is energy
  // that was never in the source (fill, noise-shaper boost). Both come from
  // `band_distortion`, computed **per bin** — a band that moves energy from
  // one line to another has the right total and is still wrong, and only the
  // per-bin split sees it: the emptied line scores a deficit and the invented
  // one an excess.
  //
  // Dividing by `thr` is what folds in ATH and spreading, since the masking
  // threshold is already their product — the same normalisation `x_mask`
  // uses, so the two measures are on one scale and the weights downstream
  // keep their meaning.
  // Fullness and Bleedless are combined here, each through its *own*
  // compression law, because the two do not move on comparable scales.
  //
  // That is the part a linear weight cannot express, and measuring proved it:
  // pricing bleed at 1.5x, 3x and 6x the deficit produced -2.43 / -2.33 /
  // -2.37 dB of bleedless — no trend at all. A scale factor inside one shared
  // law just re-ranks candidates that were already ordered the same way.
  // Giving the two different *exponents* does change the shape: with
  // P_BLEED > P_FULL, a small excess costs about what it did, while a large
  // one grows far faster than the same amount of deficit would. Missing
  // content degrades gracefully — it sounds like less of the music. Added
  // content does not: it is uncorrelated with the signal, so it does not hide
  // under it, and past a point it stops being noise and becomes a separate
  // thing the listener hears.
  //
  // Combined with max(), which is the original intent — take the worse of the
  // two — and is only meaningful once each side is on its own scale.
  //
  // The result is converted back through the inverse of `noise_loudness` so
  // the value handed downstream is still a mask-relative ratio: prominence,
  // the noise-fill blend, the hole surcharge and lambda_R all keep the units
  // they were tuned in.
  static const bool fb_off = [] {
    const char *e = std::getenv("RMPS_NO_FB");
    return e && std::atoi(e) != 0;
  }();
  if (!fb_off && deficit_e >= 0.0) {
    static const double p_full = [] {
      if (const char *e = std::getenv("RMPS_FB_P_FULL")) {
        const double v = std::atof(e);
        if (v > 0.0 && v <= 2.0)
          return v;
      }
      return LOUDNESS_EXP;          // 0.25 — unchanged from the old measure
    }();
    static const double p_bleed = [] {
      if (const char *e = std::getenv("RMPS_FB_P_BLEED")) {
        const double v = std::atof(e);
        if (v > 0.0 && v <= 2.0)
          return v;
      }
      return 0.60;
    }();
    const double x_full = deficit_e / thr;
    const double x_bleed = excess_e / thr;
    const double l_full = std::pow(1.0 + x_full, p_full) - 1.0;
    const double l_bleed = std::pow(1.0 + x_bleed, p_bleed) - 1.0;
    const double l = std::max(l_full, l_bleed);
    x_phase = std::pow(1.0 + l, 1.0 / LOUDNESS_EXP) - 1.0;
  }

  // Below 200 Hz the cost stays *linear* in the noise-to-mask ratio instead of
  // taking the compressive loudness law. The compression exists to model how
  // noise in many bands sums into one perceived loudness; for the bass, which
  // this encoder deliberately treats as must-be-transparent, it has the wrong
  // effect — x^0.25 flattens a 18 dB error down to a factor of 2.8, so no
  // weight small enough to be sane can stop the search trading bass precision
  // for mid-band precision. (Measured: weights of 8, 32 and 128 all left
  // mixed_music at 192 kbps ~12 dB down, because the LF bands are pinned at
  // the maximum scalefactor and can only be helped by a finer *global gain* —
  // a choice the flattened cost never justifies.) Linear here restores the
  // old `D · 1e15` branch's intent while keeping the band inside the model:
  // it still answers to the masking threshold, and it still costs nothing
  // when the noise is gone.
  // Prominence weight: how much of what the listener is actually hearing lives
  // in this band. The old NMR term multiplied by `e_band` outright, which is
  // why the baseline poured bits into the bass — a low band carrying 1e14 of
  // energy outscored everything else by orders of magnitude. Dropping it
  // entirely costs ~12 dB of segSNR on program material at 192 kbps, so the
  // effect is real; but linear-in-energy is far too strong to be a perceptual
  // statement. A compressive power of the band's share of the loudest band
  // keeps the ordering ("noise in a prominent band is more damaging") at a
  // sane strength.
  const double prominence =
      (PROMINENCE_EXP > 0.0)
          ? std::pow(std::max(e_band_b, 1.0) / std::max(e_ref, 1.0),
                     PROMINENCE_EXP)
          : 1.0;
  const double p_phase = weight_eff * prominence *
                         (is_below_200 ? x_phase : noise_loudness(x_phase));

  // Smoothly blend the two regimes by the noise-fill weight. A band whose
  // tonality drifts near the boundary now slides between regimes instead of
  // snapping, removing the frame-to-frame cost-mode flip that flickered the
  // HF as scattered holes.
  double perceptual = (1.0 - w_nf) * p_phase + w_nf * p_level;

  // Spectral-hole guard: a band that carried energy and is coded to silence.
  // Charged separately from the NMR because a gap is audible as roughness
  // even when noise of the same power would sit under the mask — and because
  // noise-like bands lose the most from it, hence the noise-likeness boost.
  // This applies to the Side channel too. Exempting it looks right — a
  // zeroed Side band narrows the image rather than leaving a gap, since Mid
  // still carries that region — but that reasoning only holds for correlated
  // material. On decorrelated content E[M] ≈ E[S], and dropping Side takes
  // real content out of L and R: measured +6.2 points of hole rate on
  // stereo_decorrelated at 64 kbps with the exemption in place.
  (void)side_channel;
  // Every band above the bass, not just the HF ones: the measured holes on
  // transient material sit at 345 Hz - 2.4 kHz, which the old `sfb >= 11`
  // (~5 kHz) gate never covered.
  if (sfb >= 1 && E_recon_b == 0.0 && e_band_b > 0.0) {
    const double noise_likeness = 1.0 - std::clamp(tonality_b, 0.0, 1.0);
    perceptual *= 1.0 + hole_surcharge * (1.0 + 3.0 * noise_likeness);
  } else if (side_channel && E_recon_b > 0.0 && E_recon_b < 0.50 * e_band_b) {
    // Soft attenuation (-6 dB to -36 dB) on Side channel:
    // Preserves stereo image continuity & avoids black-hole artifacts while saving bits.
    perceptual *= 0.88;
  }

  // HF over-shoot: the band was coded louder than the source. Charged only
  // upward, only above the dead-band, and only in the high region (ramped in
  // so there is no step at the gate). See HF_OVER_WEIGHT. Note this is
  // mutually exclusive with the hole guard above — a band cannot be both
  // silent and hot — so it is added, not folded into that multiplier.
  if (sfb > HF_OVER_FIRST_SFB - HF_OVER_RAMP && E_recon_b > 0.0) {
    const double w_over =
        std::clamp(static_cast<double>(sfb - (HF_OVER_FIRST_SFB -
                                              HF_OVER_RAMP)) /
                       HF_OVER_RAMP,
                   0.0, 1.0);
    // Allowance = source level plus the dead-band. A band that was dark in
    // the source (e_band ≈ 0) gets no allowance: everything coded there is
    // invented, which is the strongest case this term exists for.
    const double excess = E_recon_b - e_band_b * HF_OVER_DEADBAND;
    if (excess > 0.0)
      perceptual +=
          HF_OVER_WEIGHT * w_over * weight_eff * noise_loudness(excess / thr);
  }

  // --- Bleed constraint -------------------------------------------------
  //
  // Added content must stay inaudible. Missing content sometimes cannot be
  // afforded, so a deficit is priced and traded like anything else; energy
  // that was never in the source is different in kind — it is uncorrelated
  // with the music, so it does not hide under it, and there is never a
  // budget argument for fabricating it. The masking threshold is exactly the
  // line for "inaudible", so the constraint is `excess / thr <= limit`.
  //
  // Written as a barrier rather than a rejection. Rejecting candidates
  // outright can empty a band's cache and drop the granule to the fallback
  // path, which costs far more than the bleed did; a barrier steep enough
  // that any non-bleeding alternative wins is the same thing in practice and
  // cannot make the search infeasible. Coarser quantization always bleeds
  // less (more bins go to zero), so an admissible candidate always exists.
  //
  // This is the shape that has worked twice in this encoder — the noise
  // shaper's energy cap and the fill's admissibility conditions both moved
  // real ground, while every attempt to *re-price* the same effects measured
  // inert. Constraints change what the search may choose; weights only
  // reorder what it was going to choose anyway.
  static const double bleed_limit = [] {
    if (const char *e = std::getenv("RMPS_BLEED_LIMIT")) {
      const double v = std::atof(e);
      if (v >= 0.0)
        return v;
    }
    return 0.0;   // 0 = off
  }();
  // Only bands that are *net adding* are constrained. Rounding a coefficient
  // up is not fabrication — it is the nearest quantizer level, and half of
  // ordinary rounding goes that way — so a band whose excess is smaller than
  // its deficit is just being coded, not invented. Constraining those too was
  // measured and is what made the trade so steep: it forces systematic
  // rounding-down and bleeds fullness everywhere (limit 1.0 bought 0.43 dB of
  // bleedless for 4.6 dB of fullness). Fabrication is the case where the
  // excess *dominates*, which is exactly what the noise shaper's repayment
  // overshoot and the fill's injection produce.
  static const bool net_only = [] {
    const char *e = std::getenv("RMPS_BLEED_ALL");
    return !(e && std::atoi(e) != 0);
  }();
  const bool net_adding = !net_only || (excess_raw > deficit_e);
  if (bleed_limit > 0.0 && excess_raw > 0.0 && net_adding) {
    static const double barrier = [] {
      if (const char *e = std::getenv("RMPS_BLEED_BARRIER")) {
        const double v = std::atof(e);
        if (v > 0.0)
          return v;
      }
      return 50.0;
    }();
    const double x_bleed = excess_raw / thr;
    if (x_bleed > bleed_limit)
      perceptual += barrier * weight_eff *
                    noise_loudness(x_bleed - bleed_limit);
  }

  if (waveform_reg > 0.0)
    perceptual += waveform_reg * weight_eff * noise_loudness(x_abs);

  // --- Fine-structure penalty (additive) ---------------------------------
  //
  // `nf_structure_weight` scores texture by moving a band between the fill
  // leg and the waveform leg, and measured inert: sfm_ratio moved by <= 0.4 dB
  // on every band at 48/64/96 kbps and the harness was 0 / 0 on every metric —
  // the same fate as the density gate before it, and as every other blend
  // re-weighting in this cost. So price the texture loss directly, as its own
  // additive term, on noise-like bands only (the fill's own noise weight, so
  // tonal bands and the low spectrum are untouched): the dB by which the
  // candidate is peakier than the source, beyond a 3 dB allowance.
  //
  // **Measured too small to matter; default 0.** At W = 1 / 4 sfm_ratio moved
  // by under 1 dB and traded bands against each other (avicii@64k 8-11 kHz
  // -5.58 -> -5.01 dB, but 11-14 kHz -35.0 -> -36.3; witchqueen@48k 11-14 kHz
  // lost 4 dB of *level*). Harness at W = 1: hole_rate 4 better / 0 worse,
  // hf_balance 2 / 0, segSNR_p05 1 / 3. The defect is -13 dB (48 kbps, 5-11 kHz)
  // to -47 dB (11-14 kHz), and it is the bit budget, not the cost function:
  // flatness needs non-zero lines and the HF is already all +-1 at this rate
  // (see the memory note on exhausted HF bits).
  // RMPS_NF_SFM_W sets the weight; 0 disables it.
  {
    static const double sfm_w = [] {
      if (const char *e = std::getenv("RMPS_NF_SFM_W")) {
        const double v = std::atof(e);
        if (v >= 0.0) return v;
      }
      return 0.0;
    }();
    if (sfm_w > 0.0 && nf_sfm_db < 1e8 && !granule_is_transient &&
        !is_pre_window) {
      const double w_noise = noise_fill_weight(sfb, tonality_b, nf_first);
      const double loss_db = std::max(0.0, -nf_sfm_db - 3.0);
      if (w_noise > 0.0 && loss_db > 0.0)
        perceptual += sfm_w * w_noise * weight_eff * (loss_db / 10.0);
    }
  }

  // Fidelity reward: closeness to the source, priced independently of the
  // mask. Unlike every other term here this one is not asking "is the error
  // audible" — it is the plain statement that a smaller difference from the
  // source is never worse, so the search should prefer it whenever two
  // candidates are otherwise tied.
  //
  // Written as a saturating *penalty* rather than a literal negative reward:
  //
  //     R(x) = x / (1 + x),   x = D_b / e_ref
  //
  // which is `1 − 1/(1+x)`, i.e. the reward `−1/(1+x)` plus a constant. The
  // band count per granule is fixed at 21, so the constant shifts every
  // candidate equally and the ranking is identical — but the cost stays
  // non-negative and on the same scale as its neighbours, which the frontier
  // ranking and the debug output both assume.
  //
  // The shape is the point. Near x = 0 it is ≈ x, so it keeps a *linear*
  // gradient exactly where the compressive loudness law has flattened to
  // nothing — that is the regime where "inaudible but more accurate" lives,
  // and where the existing `waveform_reg` leg contributes almost no slope.
  // It saturates at 1, so a badly-broken band cannot use this term to outbid
  // the masking model: the reward is for getting *close*, not a second
  // opinion on how bad far is.
  if (SRC_DIFF_REWARD > 0.0)
    perceptual +=
        SRC_DIFF_REWARD * weight_eff * (x_abs / (1.0 + std::max(x_abs, 0.0)));

  // λ_R arrives already scaled into loudness units (see LAMBDA_LOUDNESS_SCALE
  // at its caller) so that the frontier ranking can use the same number.
  const double bit_cost = lambda_R * static_cast<double>(bits_est);
  return perceptual + bit_cost;
}

/// One DP state: the cheapest path reaching a given cumulative-rate bucket.
struct DpCell {
  double cost = 0.0; ///< cumulative hybrid cost (perceptual + λ_R · marginal)
  int bits = 0;      ///< cumulative *total* rate — what the budget sees
  int mbits = 0;     ///< cumulative *marginal* rate — what `cost` charged for
  bool ok = false;
};

void materialize_ix(
    const std::array<std::int8_t, NUM_SFB_LONG> &sf_path,
    const std::array<std::array<CacheEntry, SF_MAX_CANDIDATE + 1>, NUM_SFB_LONG>
        &cache,
    const std::array<int, tables::NUM_SCFB_LONG> &sfb_table, int *ix_out) {
  std::fill(ix_out, ix_out + GRANULE_SAMPLES, 0);
  for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
    const int sf = sf_path[static_cast<std::size_t>(sfb)];
    const auto &e =
        cache[static_cast<std::size_t>(sfb)][static_cast<std::size_t>(sf)];
    const int start = sfb_table[sfb];
    for (int j = 0; j < e.ix.size; ++j) {
      ix_out[start + j] = e.ix[j];
    }
  }
}

using BandCache =
    std::array<std::array<CacheEntry, SF_MAX_CANDIDATE + 1>, NUM_SFB_LONG>;

/// The search's per-band candidate cache, cleared as a fresh local would be.
/// It is 173 kB, more than a small target's whole stack (the Daisy has about
/// 33 kB), so it lives on the heap, one per thread, built on first use.
BandCache &fresh_band_cache() {
#if defined(RMPS_NO_THREADS)
  static std::unique_ptr<BandCache> c;
#else
  thread_local std::unique_ptr<BandCache> c;
#endif
  if (!c)
    return *(c = std::make_unique<BandCache>());
  for (auto &row : *c)
    for (auto &e : row) {
      e.~CacheEntry();
      ::new (&e) CacheEntry{};
    }
  return *c;
}

} // namespace

/// Phase timing for the search (RMPS_PROF=1): where the effort goes.
struct BeamProf {
  double total = 0, cache = 0, dp = 0;
  long long calls = 0;
  ~BeamProf() {
    if (calls)
      std::fprintf(stderr,
                   "[prof] beam_search calls=%lld total=%.2fs cache=%.2fs (%.0f%%) "
                   "dp+frontier=%.2fs (%.0f%%) other=%.2fs\n",
                   calls, total, cache, 100 * cache / total, dp, 100 * dp / total,
                   total - cache - dp);
  }
};
inline BeamProf &beam_prof() { static BeamProf p; return p; }
inline bool prof_on() { static const bool on = std::getenv("RMPS_PROF") != nullptr; return on; }
using prof_clock = std::chrono::steady_clock;
inline double secs(prof_clock::time_point a) {
  return std::chrono::duration<double>(prof_clock::now() - a).count();
}

BeamResult beam_search(int max_bits, int *ix, int gr, int ch, EncoderState &s,
                       const int *gr0_sf) {
  BeamResult out{};
  struct TotalGuard {
    prof_clock::time_point t = prof_clock::now();
    ~TotalGuard() {
      if (prof_on()) { beam_prof().total += secs(t); ++beam_prof().calls; }
    }
  } total_guard;

  // Search effort (CLI --effort, via EncoderState). Effort 7 reproduces the
  // constants these replaced.
  const int gain_window = std::clamp(s.cfg_gain_window, 0, 6);
  const int sf_max = std::clamp(s.cfg_sf_max, 1, SF_MAX_CANDIDATE);
  const CompressSet COMPRESS_CANDIDATES = compress_candidates(sf_max);
  const int n_buckets = std::clamp(s.cfg_rate_buckets, 2, NUM_RATE_BUCKETS);
  std::array<DpCell, NUM_RATE_BUCKETS> dp_cur{}, dp_nxt{};
  std::array<std::array<std::int8_t, NUM_RATE_BUCKETS>, NUM_SFB_LONG> dp_sf{};
  std::array<std::array<std::int16_t, NUM_RATE_BUCKETS>, NUM_SFB_LONG>
      dp_prev{};

  BandCache &cache = fresh_band_cache();
  auto &cod = s.side_info.gr[gr].ch[ch].tt;
  const auto &band_table =
      tables::scale_fact_band_index[s.mpeg.samplerate_index];

  const bool is_side =
      (s.wave.channels == 2 && ch == 1 && (s.mpeg.mode_ext & 2));
  const auto &weights = post::perceptual_weights(s.mpeg.samplerate_index,
                                                 s.wave.samplerate, is_side);

  // 3D masking thresholds for this (ch, gr) — precomputed by
  // iteration_loop. Empty (zeros) if the model was bypassed; the
  // hybrid cost then degenerates gracefully to legacy SDR via α=0.
  const auto &mt = s.mask_thresholds[ch][gr];
  const double m_alpha = s.mask_params.alpha;
  if (const char *rng = std::getenv("RMPS_THR_DUMP")) {
    int lo = 0, hi = 0;
    if (std::sscanf(rng, "%d:%d", &lo, &hi) == 2 && s.dbg_frame_no >= lo &&
        s.dbg_frame_no <= hi) {
      std::fprintf(stderr, "[thr] f=%lld gr=%d ch=%d\n", s.dbg_frame_no, gr,
                   ch);
      for (int b = 0; b < 8; ++b)
        std::fprintf(stderr,
                     "      sfb%-2d e=%.4e thr=%.4e e/thr=%8.2f dB\n", b,
                     mt.e_band[b], mt.thr[b],
                     10.0 * std::log10(std::max(mt.e_band[b], 1.0) /
                                       std::max(mt.thr[b], 1.0)));
    }
  }

  const bool granule_is_transient = s.transient_flags[gr] ||
                                    s.attack_from_silence_flags[gr] ||
                                    s.pre_window_flags[gr];
  // The granule that carries pre-echo is the one *before* the attack, and
  // until now nothing told it so. `pre_window_flags` covers a run-up inside
  // the same frame, and `cross_frame_pre_window` is set on the frame that
  // *contains* the attack — by which point the granule that needed
  // protecting has already been encoded. When the attack opens a frame, the
  // exposed granule is the previous frame's last one, which sees no flag at
  // all.
  //
  // That is the gap the one-frame lookahead closes: `lookahead_attack` says
  // an attack lands in the next frame, so this frame's tail granule is a
  // run-up and must be treated as one. Measured, that gap was the whole of a
  // 6.2 dB pre-echo regression on attack_from_silence at 64 kbps (2.4 dB on
  // transient_train), which three earlier gates — per granule, per frame, and
  // "no alpha search on attack frames at all" — could not move, because none
  // of them was looking at the right granule.
  //
  // This one flag drives every pre-echo defence at once: it clamps the
  // shaping alpha to [0.05, 0.30] (`eff_nsa` below), stops the per-band alpha
  // ladder from re-opening that choice, applies TRANSIENT_WEIGHT and the HF
  // pre-window ramp in the band cost, and disables both the noise-fill
  // injection and the zero-fill boost — extra HF right before an attack is
  // exactly what decodes as pre-echo hiss.
  const bool next_frame_attack =
      s.lookahead_valid && s.lookahead_attack &&
      (gr == s.mpeg.granules_per_frame - 1);
  if (std::getenv("RMPS_LA_DEBUG"))
    std::fprintf(stderr, "[la] gr=%d valid=%d attack=%d demand=%.3f fires=%d\n",
                 gr, static_cast<int>(s.lookahead_valid),
                 static_cast<int>(s.lookahead_attack), s.lookahead_demand,
                 static_cast<int>(next_frame_attack));
  const bool is_pre_window = s.pre_window_flags[gr] ||
                             (gr == 0 && s.cross_frame_pre_window) ||
                             next_frame_attack;

  // Per-granule adaptive noise shaping alpha based on spectral tonality &
  // transient metrics:
  double mean_tonality = 0.0;
  for (int b = 0; b < NUM_SFB_LONG; ++b) {
    mean_tonality += mt.tonality[b];
  }
  mean_tonality /= static_cast<double>(NUM_SFB_LONG);

  double eff_nsa = s.cfg_noise_shaping_alpha;
  if (is_pre_window) {
    // Pre-window (immediately preceding an attack): reduce frequency-domain
    // shaping alpha so quantization error doesn't diffuse into the pre-attack zone.
    eff_nsa = std::clamp(eff_nsa * 0.35, 0.05, 0.30);
  } else if (granule_is_transient) {
    // Transient granule: moderate shaping to prevent pre-echo spill
    eff_nsa = std::clamp(eff_nsa * 0.70, 0.15, 0.50);
  } else {
    // Highly tonal spectrum -> scale down shaping to prevent partial smearing;
    // Noise-like / complex spectrum -> scale up shaping to maximize masking
    // benefit.
    eff_nsa = std::clamp(
        s.cfg_noise_shaping_alpha * (1.35 - 0.85 * mean_tonality), 0.15, 0.90);
  }
  // Master off-switch for the frequency-domain noise shaper (RMPS_NO_NSA=1).
  // Forces alpha to 0 everywhere, which makes `quantize_band`'s target the
  // coefficient itself (`target = x - alpha*err` with alpha = 0) and removes
  // the per-band alpha search below along with it.
  static const bool nsa_disabled = [] {
    const char *e = std::getenv("RMPS_NO_NSA");
    return e && std::atoi(e) != 0;
  }();
  const double nsa = nsa_disabled ? 0.0 : eff_nsa;

  // Per-promotion bound for this granule (see `quantize_band`), in amplitude
  // dB over the promoted coefficient's own source value. Applied only where the
  // granule's energy *rises* — that is where an invented line lands in front of
  // something and becomes a click — and not on steady granules, where the same
  // promotions are what fills the holes.
  //
  // Why it has to be scoped, and why not by the transient flags:
  //
  // Applied everywhere at +6 dB it took avicii's pre-attack clicks from 82 to
  // 18 at 48 kbps, but cost hole_rate 1 better / 7 worse on the harness
  // (hf_hiss_decay -8 points, transient_train -6): the promotions that click
  // and the promotions that fill holes are the same 6-12 dB class, so no dB
  // value separates them — +12 dB left 79 of the 82 clicks, +18 left 72.
  //
  // What does separate them is the time structure. Measured on the source, the
  // granules the clicks come from rise by >= 2x within a 192-sample sub-window
  // in 70 % of cases, against 23 % of all granules; steady noise almost never
  // does. The transient flags cannot be used for this: scoping the bound to
  // them left 72 of the 82 clicks, because the gate ignores any sub-window
  // under `ABS_ENERGY_FLOOR` (about -43 dBFS RMS) and these clicks sit in quiet
  // passages below it. So the rise is read from the gate's sub-window energies
  // directly, with a floor low enough not to go blind in quiet material.
  //
  //   flagged attack / run-up granule   -> RMPS_NSA_PROMO_ATK_DB (0 dB)
  //   everything else                   -> RMPS_NSA_PROMO_DB (+6 dB)
  // A negative dB disables that tier.
  //
  // **Scoping measured and rejected.** The reasoning above predicted the rise
  // test would keep the click reduction and drop the hole cost. It kept the
  // hole cost down (harness neutral) and lost the click reduction: avicii's
  // pre-attack clicks at 48 / 64 / 96 kbps were 82 / 40 / 13 before, 18 / 22 / 6
  // with the bound everywhere, 72 / 35 / 9 with it scoped to rising granules and
  // 72 / 35 / 9 scoped to flagged ones. The promotions that click are in
  // granules neither test marks — most likely the quiet granule *before* the
  // rise, whose own window never sees the attack but whose invented line rings
  // straight into the silence in front of it. So the bound applies everywhere
  // by default and the hole cost is the price: hole_rate 1 better / 7 worse
  // (hf_hiss_decay about -8 points, transient_train about -6), against
  // segSNR_p05 8 better / 3 worse. RMPS_NSA_RISE=2 restores the rise scope.
  {
    static const double promo_db = [] {
      if (const char *e = std::getenv("RMPS_NSA_PROMO_DB")) return std::atof(e);
      return 6.0;
    }();
    static const double promo_atk_db = [] {
      if (const char *e = std::getenv("RMPS_NSA_PROMO_ATK_DB"))
        return std::atof(e);
      return 0.0;
    }();
    static const double rise_t = [] {
      if (const char *e = std::getenv("RMPS_NSA_RISE")) {
        const double v = std::atof(e);
        if (v >= 0.0) return v;
      }
      return 0.0;   // 0 = no scope, bound everywhere
    }();
    // Rise within this granule, including its first sub-window against the
    // last one before it. The floor keeps near-silence ratios out without
    // repeating the gate's -43 dBFS blind spot: 1e3 over 192 int16 samples is
    // about -63 dBFS RMS.
    constexpr double RISE_FLOOR = 1.0e3;
    const auto &se = s.sub_window_energies[gr];
    double prev_e = (gr > 0) ? s.sub_window_energies[gr - 1][2]
                             : s.prev_frame_last_sub_energy;
    double rise = 1.0;
    for (int w = 0; w < 3; ++w) {
      const double e = se[static_cast<std::size_t>(w)];
      if (e > RISE_FLOOR)
        rise = std::max(rise, e / std::max(prev_e, RISE_FLOOR));
      prev_e = e;
    }
    double db = -1.0;
    if (is_pre_window || granule_is_transient)
      db = promo_atk_db;
    else if (rise_t <= 0.0 || rise >= rise_t)
      db = promo_db;
    g_nsa_promo_ratio = (db < 0.0) ? 0.0 : std::pow(10.0, db / 20.0);
    g_granule_rising = is_pre_window || granule_is_transient ||
                       (rise >= (rise_t > 0.0 ? rise_t : 2.0));
  }

  // Two unit conversions, both applied once here so every consumer (the band
  // cost and the frontier ranking) uses the same number:
  //  · /2.6  — λ_R was tuned against the old lower-bound rate term; the RD
  //            term now charges `BandRate::marginal`, measured 2.0× (64 kbps)
  //            to 3.1× (320 kbps) larger.
  //  · ×1e-11 — λ_R's legacy magnitude (5e8 at 64 kbps) belonged to a cost
  //            that ran to 1e15. Against the loudness cost it would swamp
  //            everything; this brings λ_R·bits to ≈1 for a 200-bit band at
  //            64 kbps, i.e. the same order as a band audibly over its mask.
  constexpr double LAMBDA_RATE_UNIT = 1.0 / 2.6;
  const double m_lambda_R = s.mask_params.lambda_R * s.lambda_granule_scale *
                            LAMBDA_RATE_UNIT * LAMBDA_LOUDNESS_SCALE;
  const double m_waveform_reg = s.mask_params.waveform_reg;
  const double gr_energy_contrast = s.energy_contrast[gr];
  // Reference level for the α=0 (waveform) leg of the band cost: the loudest
  // band in this granule. Normalising by a granule-wide constant keeps that
  // leg proportional to the raw distortion, which is what it was before the
  // cost moved into loudness units.
  double e_ref = 1.0;
  for (int b = 0; b < NUM_SFB_LONG; ++b)
    e_ref = std::max(e_ref, mt.e_band[b]);
  if (std::getenv("RMPS_EREF_DEBUG")) {
    double emin = 1e300, esum = 0.0;
    for (int b = 0; b < NUM_SFB_LONG; ++b) {
      emin = std::min(emin, std::max(mt.e_band[b], 0.0));
      esum += mt.e_band[b];
    }
    double wnf_sum = 0.0;
    for (int b = 0; b < NUM_SFB_LONG; ++b)
      wnf_sum += noise_fill_weight(b, mt.tonality[b]);
    std::fprintf(stderr,
                 "[eref] e_ref=%.4g e_min=%.4g e_mean=%.4g thr0=%.4g "
                 "alpha=%.3f wnf_mean=%.3f\n",
                 e_ref, emin, esum / NUM_SFB_LONG, mt.thr[0], m_alpha,
                 wnf_sum / NUM_SFB_LONG);
  }

  // Adapt hf_boost based on bitrate & granule HF energy fraction:
  // Measure ratio of high-frequency energy (SFBs 17-21, >7 kHz) vs total energy
  double tot_granule_e = 1e-9;
  double hf_granule_e = 0.0;
  for (int b = 0; b < NUM_SFB_LONG; ++b) {
    tot_granule_e += mt.e_band[b];
    if (b >= 17)
      hf_granule_e += mt.e_band[b];
  }
  const double r_hf = hf_granule_e / tot_granule_e;

  double eff_hf_boost = s.hf_weight_boost;
  if (s.policy_bitr() > 0) {
    double br_factor = 1.0;
    if (s.policy_bitr() >= 128) {
      double t = static_cast<double>(s.policy_bitr() - 128) / (320.0 - 128.0);
      br_factor = 1.0 - 0.4 * t; // scales down to 0.6x at 320 kbps
    } else {
      // Let RDO naturally govern high-frequency bit allocation at low bitrates
      // rather than forcing an artificial boost.
      br_factor = 1.0;
    }
    double hf_energy_factor = 1.0;
    if (r_hf < 0.02) {
      // Very low HF content -> scale down boost to save bits for LF/mid
      hf_energy_factor = std::max(0.25, r_hf / 0.02);
    } else if (r_hf >= 0.05) {
      hf_energy_factor = 1.0 + std::min(0.5, (r_hf - 0.05) * 2.0);
    }
    // HF emphasis floor on granules the gate saw as HF transients (hi-hat,
    // cymbal hits). Steady granules get at most the r_hf-driven factor, so
    // this is weight on an attack's top that the material around it does not
    // get. RMPS_HF_TRANS_FACTOR; default unchanged.
    static const double hf_trans_factor = [] {
      if (const char *e = std::getenv("RMPS_HF_TRANS_FACTOR")) {
        const double x = std::atof(e);
        if (x > 0.0 && x < 10.0) return x;
      }
      return 1.35;
    }();
    if (s.hf_transient_flags[gr]) {
      hf_energy_factor = std::max(hf_energy_factor, hf_trans_factor);
    }
    eff_hf_boost =
        std::max(1.0, s.hf_weight_boost * br_factor * hf_energy_factor);
  }

  // Shared low-bitrate ramp in [0,1]: 0 at ≥128 kbps, 1 at 32 kbps. The
  // black-spot fill scales with it — at starved rates the dead zone eats
  // far more of the spectrum, so the fill may reach deeper and closer to
  // the source level. Transparency is preserved regardless: every fill
  // path stays capped at the band's actual energy deficit.
  double low_rate_t = 0.0;
  if (s.policy_bitr() > 0 && s.policy_bitr() < 128)
    low_rate_t =
        std::clamp((128.0 - s.policy_bitr()) / (128.0 - 32.0), 0.0, 1.0);

  // Lowest band the noise fill may reach at this rate (see nf_first_sfb).
  const int nf_first = nf_first_sfb(s.policy_bitr());

  // Fullness/Bleedless objective instead of the NMR/SDR blend. Opt-in while
  // it is being evaluated against the existing cost.
  static const bool fb_cost = [] {
    const char *e = std::getenv("RMPS_FB_COST");
    return e && std::atoi(e) != 0;
  }();

  // Local mutable copies of weights and thresholds for fundamental and harmonic
  // adjustments
  auto adj_weights = weights;
  auto adj_thr = mt.thr; // Already includes harmonic threshold boosts from the
                         // masking model

  // Bands carrying a resolved harmonic series (see the detector below).
  std::array<bool, NUM_SFB_LONG> band_is_comb{};
  int fund_bin = -1;
  std::int32_t max_val = 0;
  // Scan bins 4 to 120 (roughly 150 Hz to 4.5 kHz at 44.1k)
  for (int i = 4; i < 120; ++i) {
    if (s.xrabs[i] > max_val) {
      max_val = s.xrabs[i];
      fund_bin = i;
    }
  }

  if (fund_bin >= 0 && max_val > scale::magnitude(1000)) {
    auto sfb_containing = [&](int bin) {
      for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        if (bin >= band_table[sfb] && bin < band_table[sfb + 1]) {
          return sfb;
        }
      }
      return -1;
    };

    // Fractional fundamental — see psy::refine_peak_bin. With an integer f0
    // bin the predicted partials drift out from under the real ones as k
    // grows, so this discount lands on empty spectrum instead of the partials
    // it is meant to describe.
    double f0_bin = fund_bin;
    if (fund_bin > 0 && fund_bin + 1 < GRANULE_SAMPLES) {
      const double pa = s.xrabs[static_cast<std::size_t>(fund_bin - 1)];
      const double pb = s.xrabs[static_cast<std::size_t>(fund_bin)];
      const double pc = s.xrabs[static_cast<std::size_t>(fund_bin + 1)];
      const double den = pa - 2.0 * pb + pc;
      if (den != 0.0)
        f0_bin = fund_bin + std::clamp(0.5 * (pa - pc) / den, -0.5, 0.5);
    }
    for (int k = 2; k <= 8; ++k) {
      int harm_bin = static_cast<int>(k * f0_bin + 0.5);
      if (harm_bin >= GRANULE_SAMPLES)
        break;
      (void)sfb_containing(harm_bin);
      // The bands carrying partials used to get their perceptual weight
      // *discounted* here (0.75 on the partial's band, 0.85 either side), on
      // the theory that a fused harmonic series hides its own partials. That
      // is true of the fusion percept and false of the artefact: a partial
      // coded away does not vanish quietly, it stops the series partway up
      // and the timbre changes character mid-spectrum. Now that the harmonic
      // map actually points at the partials (see the fractional f0 above),
      // that discount aims straight at the content it was hurting.
    }

    // --- Harmonic-comb detection, per band ---
    //
    // A band whose energy sits on the partials of the detected series is
    // *structured*, however "noisy" its spectral flatness looks. The SFM
    // tonality that gates the noise-fill region cannot see this: several
    // partials with gaps between them inside one scalefactor band read as
    // flat, so a continuing harmonic series above ~5 kHz gets classified
    // noise-like and replaced with level-matched noise. Measured on a 220 Hz
    // saw at 64 kbps, partials 35–38 (7.7–8.4 kHz) came out 14–17 dB down —
    // the series simply stops partway up, which is audible as the timbre
    // changing character mid-spectrum however high the band is.
    //
    // So: measure what fraction of each band's energy lies within ±1 bin of a
    // predicted partial. Above the threshold the band is a comb and is held
    // out of the noise-fill regime — it gets coded, not synthesised.
    constexpr double COMB_ENERGY_FRAC = 0.35;
    constexpr int COMB_MIN_PARTIALS = 2;
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
      const int lo = band_table[sfb], hi = band_table[sfb + 1];
      const int k_lo = std::max(2, static_cast<int>(lo / f0_bin));
      const int k_hi = static_cast<int>(hi / f0_bin) + 1;
      double e_comb = 0.0;
      int n_partials = 0;
      for (int k = k_lo; k <= k_hi; ++k) {
        const int c = static_cast<int>(k * f0_bin + 0.5);
        if (c < lo || c >= hi)
          continue;
        ++n_partials;
        for (int j = std::max(lo, c - 2); j < std::min(hi, c + 3); ++j) {
          const double v = s.xrabs[static_cast<std::size_t>(j)];
          e_comb += v * v;
        }
      }
      if (n_partials < COMB_MIN_PARTIALS)
        continue;
      double e_tot = 0.0;
      for (int j = lo; j < hi; ++j) {
        const double v = s.xrabs[static_cast<std::size_t>(j)];
        e_tot += v * v;
      }
      if (e_tot > 0.0 && e_comb / e_tot >= COMB_ENERGY_FRAC)
        band_is_comb[static_cast<std::size_t>(sfb)] = true;
    }
  }

  // Per-band dead-zone relief factors for the transparent zero fill
  // (--hf-zero-boost). Graded by the noise-fill weight so tonal bands get
  // no relief (a lone |ix|=1 next to a coded tone is audible as noise) and
  // the region boundary tapers smoothly instead of flickering. Transient
  // and pre-window granules are never filled — extra HF right before an
  // attack is exposed as pre-echo hiss.
  std::array<double, NUM_SFB_LONG> zb_factor;
  zb_factor.fill(1.0);
  double zb_factor_hf = 1.0; // implicit-sf0 region above the top SFB
  if (s.cfg_hf_zero_boost_db > 0.0 && !fill_disabled() &&
      !granule_is_transient && !is_pre_window) {
    // Up to 2x the configured relief window at 32 kbps (capped at the
    // CLI's own 24 dB limit): the compressive curve then lifts deeper
    // coefficients before its saturation cap kicks in.
    const double zb_db =
        std::min(s.cfg_hf_zero_boost_db * (1.0 + low_rate_t), 24.0);
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
      const double w = band_is_comb[static_cast<std::size_t>(sfb)]
                           ? 0.0
                           : noise_fill_weight(sfb, mt.tonality[sfb],
                                               nf_first_sfb(s.policy_bitr()));
      if (w > 0.0)
        zb_factor[sfb] = std::pow(10.0, zb_db * (w * w) / 20.0);
    }
    zb_factor_hf = std::pow(10.0, zb_db / 20.0);
  }
  // Flatten the zero-fill companding curve as the rate starves: exponent
  // 1.0 (half-the-depth lift) at ≥128 kbps down to 0.5 at 32 kbps, so
  // deep dead-zone coefficients still reach promotion. The per-band
  // deficit budget continues to bound the total — the reach grows, the
  // ceiling does not.
  const double zb_fill_exp = 1.0 - 0.5 * low_rate_t;

  // Hole surcharge, scaled by how starved the rate is: full strength at
  // 32 kbps, base at 96 kbps and above.
  const double hole_t =
      (s.policy_bitr() > 0)
          ? std::clamp((96.0 - s.policy_bitr()) / (96.0 - 32.0), 0.0, 1.0)
          : 0.0;
  const double eff_hole_surcharge = HOLE_SURCHARGE * (1.0 + 2.0 * hole_t);

  // Pre-extract band slices.
  std::array<const std::int32_t *, NUM_SFB_LONG> xr_bands{};
  std::array<const std::int32_t *, NUM_SFB_LONG> xrabs_bands{};
  std::array<std::int32_t, GRANULE_SAMPLES> xrabs_ext{};
  std::array<const std::int32_t *, NUM_SFB_LONG> xr_ext_bands{};
  std::array<const std::int32_t *, NUM_SFB_LONG> xrabs_ext_bands{};
  std::array<bool, NUM_SFB_LONG> has_ext_hf{};
  has_ext_hf.fill(false);
  std::array<const int *, NUM_SFB_LONG> prev_bands{};
  std::array<int, NUM_SFB_LONG> band_lens{};
  // The previous granule's indices feed the quantizer's zero hysteresis, and
  // are only line-comparable when both granules are LONG — a short granule
  // interleaves three sub-windows, so line i is not the same frequency.
  const int *prev_base =
      (s.block_type[ch][gr] == BLOCK_LONG && s.prev_l3_enc_long[ch] &&
       s.prev_l3_enc_mode_ext == s.mpeg.mode_ext)
          ? s.prev_l3_enc[ch].data()
          : nullptr;
  // Rate window for the zero hysteresis, measured on the harness at
  // 32/48/64/96/128/192/320.
  //
  // Its useful range is narrow and both edges have a mechanism. Above about
  // 96 kbps there is nothing to fix — the decode already moves 4.1 dB against
  // the source's 3.4 dB, i.e. it tracks the material, and the lines still
  // landing on the 0/1 boundary up there are the bottom of the noise floor,
  // not partials; making those sticky is pure cost (two_tone_polyphony
  // segSNR_p05 -0.77 at 128, -0.51 at 192).
  //
  // Below about 40 kbps it turns harmful for the opposite reason: the granule
  // is so starved that a line kept alive is not competing with another line
  // in the same band but with the *bandwidth* — the deadband is neutral in
  // line count, not in Huffman cost, and a low-frequency line costs more than
  // the high one it displaces. Measured at 32: mixed_music hf_balance
  // -7.69 -> -13.54 dB and sweep segSNR -2.44.
  //
  // In between it pays, coherently rather than by cell: at 48-96 the whole
  // testset moves one way, segSNR +0.19 to +0.23 on mixed_music, both stereo
  // cases, broadband_noise, harmonic_tone and two_tone_polyphony.
  const double hyst_scale = [&] {
    const double br = s.policy_bitr();
    if (br <= 0.0) return 1.0;                 // rate unknown: leave it on
    if (br < 36.0 || br > 128.0) return 0.0;
    if (br < 48.0) return (br - 36.0) / 12.0;  // ramp in
    if (br <= 96.0) return 1.0;
    return (128.0 - br) / 32.0;                // ramp out
  }();

  // --- Band hole guard ---
  //
  // The 黒点 is not "a hole", it is a hole that disagrees with its temporal
  // neighbours. A band left empty in the same place granule after granule
  // reads as a notch — a spectral shape — and is clean in MDCT terms: both
  // overlap neighbours are zero too, so TDAC has nothing to cancel. A band
  // emptied for *one* granule while it is coded on both sides loses one half
  // of a cancelling pair, and the overlap-add leaks aliasing on top of the
  // energy it dropped. That is what shows as a black spot in white noise or
  // in the middle of a sustained harmonic.
  //
  // Measured before this (RMPS_HOLE_ISO, 30 s of real material): `hf_rdo`
  // made none of them; the synthetic testset has essentially none (so the
  // harness cannot see this at all); and on real music 40-60 % of the isolated
  // holes in Mid above 6 kHz and in Side at 2-6 kHz sat on a *steady* source —
  // within 3 dB of both neighbours — i.e. the band's own quantization decision
  // flipping, not the material changing.
  //
  // So: when a band was coded last granule, is active now, and the source is
  // steady, an all-zero candidate is not admissible for it — provided a
  // non-empty one exists (checked per compress in the DP below), so this can
  // never make a granule infeasible. It stops a hole from *starting* on a
  // steady source, which is exactly the one-granule case: a hole that starts
  // because the source genuinely dropped is still allowed, and one that
  // already exists may continue, so notches are untouched.
  //
  // A constraint, not a re-weighting — the shape that works in this encoder.
  // RMPS_HOLE_GUARD: 0 off, 1 all channels (default), 2 not on Side.
  // RMPS_HOLE_GUARD_DB: the steadiness window, default 3 dB.
  std::array<bool, NUM_SFB_LONG> hole_guard{};
  std::array<double, NUM_SFB_LONG> prev_erec{};
  {
    static const int guard_mode = [] {
      if (const char *e = std::getenv("RMPS_HOLE_GUARD")) {
        const int v = std::atoi(e);
        if (v >= 0 && v <= 2) return v;
      }
      return 1;
    }();
    static const double steady_db = [] {
      if (const char *e = std::getenv("RMPS_HOLE_GUARD_DB")) {
        const double v = std::atof(e);
        if (v > 0.0) return v;
      }
      return 3.0;
    }();
    const bool role_ok =
        guard_mode == 1 ||
        (guard_mode == 2 && !(s.mpeg.mode_ext == 2 && ch == 1));
    if (prev_base && guard_mode > 0 && role_ok) {
      double emax = 0.0;
      for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb)
        emax = std::max(emax, mt.e_band[static_cast<std::size_t>(sfb)]);
      for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const auto k = static_cast<std::size_t>(sfb);
        const double e_now = mt.e_band[k];
        const double e_prev = s.prev_e_band[static_cast<std::size_t>(ch)][k];
        if (emax <= 0.0 || e_now <= emax * 1e-5 || e_prev <= 0.0)
          continue;
        if (std::abs(10.0 * std::log10(e_now / e_prev)) >= steady_db)
          continue;
        bool coded = false;
        for (int i = band_table[sfb]; i < band_table[sfb + 1]; ++i)
          if (prev_base[i] != 0) {
            coded = true;
            break;
          }
        hole_guard[k] = coded;
      }
    }
    if (prev_base)
      prev_erec = s.prev_erec_band[static_cast<std::size_t>(ch)];
    s.prev_erec_band[static_cast<std::size_t>(ch)].fill(0.0);
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb)
      s.prev_e_band[static_cast<std::size_t>(ch)]
                   [static_cast<std::size_t>(sfb)] =
          mt.e_band[static_cast<std::size_t>(sfb)];
  }

  // The two halves of the deadband are NOT worth separating at an onset —
  // measured, and it is the opposite of what the mechanism suggests.
  //
  // The reasoning was: "keep alive" is harmless at an attack, but "harder to
  // start" is not, because the previous granule is the quiet run-up, so every
  // line of the new content is a line that was zero and raising its threshold
  // delays the onset. attack_from_silence at 48 kbps did regress on pre-echo
  // (-55.98 -> -54.26 dB), so it looked like the culprit.
  //
  // Dropping the "start" half on every granule the transient machinery flags
  // (transient / pre_window / attack_from_silence / cross_frame_pre_window)
  // left attack_from_silence **bit-identical at all seven bitrates** — every
  // metric equal to two decimals — so the flags do not fire where that
  // pre-echo is decided, and the hypothesis is simply wrong. It was not free:
  // mixed_music lost segSNR_p05 -0.89 at 64 and hf_balance -0.80 at 64 /
  // -0.54 at 96. 0 better, 3 worse. Not repeated.
  for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
    const int start = band_table[sfb];
    xr_bands[sfb] = s.xr + start;
    xrabs_bands[sfb] = s.xrabs.data() + start;
    prev_bands[sfb] = prev_base ? prev_base + start : nullptr;
    band_lens[sfb] = band_table[sfb + 1] - start;
  }

  if (s.cfg_hf_extension) {
    const auto &ext_freq = s.mdct_freq_ext[ch][gr];
    for (int i = 0; i < GRANULE_SAMPLES; ++i) {
      xrabs_ext[static_cast<std::size_t>(i)] =
          std::abs(ext_freq[static_cast<std::size_t>(i)]);
    }
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
      const int start = band_table[sfb];
      xr_ext_bands[sfb] = ext_freq.data() + start;
      xrabs_ext_bands[sfb] = xrabs_ext.data() + start;
      const double lo_hz = band_table[sfb] * s.wave.samplerate / 1152.0;
      if (lo_hz <= 20000.0) {
        for (int j = 0; j < band_lens[sfb]; ++j) {
          if (xrabs_ext_bands[sfb][j] > 0) {
            has_ext_hf[sfb] = true;
            break;
          }
        }
      }
    }
  }

  // Signal power (for stats and B&B).
  double raw_signal_pow = 0.0;
  double weighted_signal_pow = 0.0;
  for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
    double sp = 0.0;
    for (int j = 0; j < band_lens[sfb]; ++j) {
      const double v = xr_bands[sfb][j];
      sp += v * v;
    }
    raw_signal_pow += sp;
    weighted_signal_pow += sp * weights[sfb];
  }
  out.raw_signal_pow = raw_signal_pow;
  out.weighted_signal_pow = weighted_signal_pow;

  // High-frequency region (sfb 21, ~16–20 kHz at 44.1k) signal power. The 21
  // transmitted scalefactor bands stop at band_table[NUM_SFB_LONG], but the
  // region above is still coded (with an implicit scalefactor of 0) by the
  // HF post-pass near the end of this function — accumulate its power so the
  // reported SDR stays honest.
  const int hf_start = band_table[NUM_SFB_LONG];
  const int hf_end = std::min(band_table[NUM_SFB_LONG + 1], GRANULE_SAMPLES);
  double hf_signal_pow = 0.0;
  for (int i = hf_start; i < hf_end; ++i) {
    const double v = s.xr[i];
    hf_signal_pow += v * v;
  }
  out.raw_signal_pow += hf_signal_pow;
  out.weighted_signal_pow += hf_signal_pow * weights[NUM_SFB_LONG - 1];

  // Seed gain via legacy bin-search.
  cod.scalefac_compress = COMPRESS_CANDIDATES[0];
  int seed_gain = bin_search_StepSize(max_bits, ix, cod, s);

  // Reserve bit headroom for the ≥16 kHz implicit-sf0 region so the splice at
  // the end of this function is not left bidding for scraps (see
  // HF21_RESERVE_*). Priced at the seed gain — the chosen gain lands within
  // ±gain_window of it, and the slack factor absorbs the difference. A silent
  // region quantizes to all zeros and reserves nothing.
  //
  // Never anywhere in a frame that contains an attack. On the attacking
  // granule itself the reservation is the wrong trade twice over — the bits
  // it takes are the ones the attack needs, and >16 kHz content is the least
  // worth defending during a hit — but gating only that granule is not
  // enough, because a granule does not spend in isolation: what a quiet
  // long-block granule does not hand back goes into the bit reservoir, which
  // is exactly where the short-block attack granule beside it draws from.
  // Gated per granule, transient_train still gave up 7.5 dB of worst-case
  // segSNR at 192 kbps (p05 −12.2 → −19.7) for 1.7 dB of >8 kHz balance;
  // the attack granules were never reserving, they were being drained.
  bool frame_has_attack = s.cross_frame_pre_window;
  for (int g = 0; g < s.mpeg.granules_per_frame; ++g)
    frame_has_attack |= s.transient_flags[static_cast<std::size_t>(g)] ||
                        s.pre_window_flags[static_cast<std::size_t>(g)] ||
                        s.attack_from_silence_flags[static_cast<std::size_t>(g)];

  int hf21_reserve = 0;
  if (hf_start < hf_end && !granule_is_transient && !is_pre_window &&
      !frame_has_attack) {
    const std::int32_t *xrabs_ptr = s.xrabs.data() + hf_start;
    if (s.cfg_hf_extension) {
      bool orig_silent = true;
      for (int j = 0; j < hf_end - hf_start; ++j) {
        if (xrabs_ptr[j] > 0) { orig_silent = false; break; }
      }
      if (orig_silent) {
        xrabs_ptr = xrabs_ext.data() + hf_start;
      }
    }
    auto q = quantize_band(xrabs_ptr, hf_end - hf_start,
                           s.steptabi[seed_gain + 127],
                           s.steptab[seed_gain + 127], /*sf=*/0, /*scale=*/0,
                           /*preflag=*/0, /*sfb=*/NUM_SFB_LONG, s.int2idx, nsa,
                           zb_factor_hf, zb_fill_exp,
                           prev_base ? prev_base + hf_start : nullptr, hyst_scale);
    if (q.ok && q.max_val > 0) {
      // Price the region by its *marginal* cost on a real granule, with and
      // without the splice. Pricing it standalone via `band_rate` cannot see
      // that splicing also moves the big_values / count1 boundary and
      // re-prices the tail, which it under-charged by ~40 % on bright
      // material: hf_hiss_decay then failed the affordability test at every
      // bitrate over a cost it was never going to pay.
      //
      // Quantize the granule at the seed gain rather than reusing what the
      // bin search left in `ix` — that is whichever gain it probed last, not
      // the one it returned.
      std::array<int, GRANULE_SAMPLES> probe{};
      quantize(probe.data(), seed_gain, s);
      std::fill(probe.begin() + hf_start, probe.begin() + hf_end, 0);
      GrInfo g_probe{};
      const int bits_without = final_bits(probe.data(), g_probe, s);
      for (int j = 0; j < hf_end - hf_start; ++j)
        probe[static_cast<std::size_t>(hf_start + j)] =
            q.ix[static_cast<std::size_t>(j)];
      const int bits_with = final_bits(probe.data(), g_probe, s);
      const int est =
          static_cast<int>((bits_with - bits_without) * HF21_RESERVE_SLACK);
      // All or nothing. A reservation smaller than what the region costs is
      // the worst of both worlds — the coded bands lose the bits and the
      // splice is still dropped for being a hundred short — and that is the
      // case a clamp produces, silently, on exactly the bright material that
      // makes the region expensive. Clamping to 32 % of the granule did this
      // on almost every granule of broadband_noise / hf_hiss_decay at
      // 192 kbps and cost 3.7 dB of worst-case segSNR for nothing.
      const double frac_t = std::clamp(
          static_cast<double>(s.policy_bitr() - HF21_FRAC_LO_KBPS) /
              (HF21_FRAC_HI_KBPS - HF21_FRAC_LO_KBPS),
          0.0, 1.0);
      const double max_frac =
          HF21_RESERVE_MAX_FRAC_LO +
          (HF21_RESERVE_MAX_FRAC_HI - HF21_RESERVE_MAX_FRAC_LO) * frac_t;
      if (est <= static_cast<int>(max_bits * max_frac))
        hf21_reserve = std::max(est, 0);
    }
  }
  // What the search itself may spend. The splice, and the temporal-rounding
  // polish that runs after it, still see the full `max_bits`.
  const int search_bits = max_bits - hf21_reserve;

  // Re-seed the gain against the budget the DP will actually have.
  //
  // The seed above was found for `max_bits`, but the DP spends
  // `search_bits - part2` — smaller by the scalefactor field and by whatever
  // the >=16 kHz region reserved. A seed found for the larger number is
  // systematically too *fine*, and `viable_gains` only spans +/-gain_window
  // around it. If no gain in that window fits the real budget, every DP
  // candidate overflows and the granule falls back to the legacy path.
  {
    const int dp_budget =
        search_bits -
        part2_length_for(COMPRESS_CANDIDATES[0], {0, 0, 0, 0}, gr);
    if (dp_budget > 0)
      seed_gain = bin_search_StepSize(dp_budget, ix, cod, s);
  }

  if (std::getenv("RMPS_HF21_DEBUG"))
    std::fprintf(stderr, "[hf21] reserve %d of %d frac %.3f eshare %.5f\n",
                 hf21_reserve, max_bits,
                 static_cast<double>(hf21_reserve) / std::max(max_bits, 1),
                 hf_signal_pow / std::max(raw_signal_pow, 1.0));

  // R-D pre-screen window.
  std::vector<int> viable_gains;
  {
    const int budget =
        search_bits - part2_length_for(COMPRESS_CANDIDATES[0], {0, 0, 0, 0}, gr);
    const int budget_max = (budget < 0) ? search_bits : budget;
    std::array<int, GRANULE_SAMPLES> tmp_ix{};
    for (int dg = -gain_window; dg <= gain_window; ++dg) {
      const int g = seed_gain + dg;
      if (g < -120 || g > 119)
        continue;
      if (quantize(tmp_ix.data(), g, s) > 8192)
        continue;
      GrInfo tmp{};
      const int bits = final_bits(tmp_ix.data(), tmp, s);
      if (bits >= 0.4 * budget_max && bits <= 1.6 * budget_max)
        viable_gains.push_back(g);
    }
    if (std::find(viable_gains.begin(), viable_gains.end(), seed_gain) ==
            viable_gains.end() &&
        seed_gain >= -120 && seed_gain <= 119)
      viable_gains.push_back(seed_gain);
    if (viable_gains.empty())
      viable_gains.push_back(seed_gain);
    std::sort(viable_gains.begin(), viable_gains.end());
    viable_gains.erase(std::unique(viable_gains.begin(), viable_gains.end()),
                       viable_gains.end());
  }

  const bool mpeg1 = (s.mpeg.version == 3);
  const std::size_t scfsi_count =
      (gr == 1 && gr0_sf != nullptr && mpeg1) ? SCFSI_PATTERNS.size() : 1u;

  // Best-so-far global tracker.
  struct Best {
    std::array<std::int8_t, NUM_SFB_LONG> sf{};
    std::array<int, GRANULE_SAMPLES> ix{};
    double cum_dist = 0.0;
    std::vector<double> dist_band; ///< hybrid cost per band (ranking)
    std::vector<double> raw_band;  ///< raw quantization noise per band
    /// `raw_band` split by the sign of the magnitude error: energy the coding
    /// failed to keep, and energy it invented. Feeds the Fullness/Bleedless
    /// report so the rate controller can target them instead of SDR.
    std::vector<double> def_band;
    std::vector<double> exc_band;
    GrInfo tmp_gi{};
    int bits = 0;
    int part2_length = 0;
    int compress = 0;
    int scale = 0;
    int preflag = 0;
    std::array<int, 4> scfsi_pat{0, 0, 0, 0};
    int gain = 0;
    /// Shaping alpha per band, as chosen by the cache. Everything downstream
    /// that re-quantizes (the ±1 polish) must use *these*, not the granule's
    /// heuristic `nsa`: quantizing a band with a different alpha than the one
    /// its cost was computed for leaves `dist_band` describing coefficients
    /// that are not the ones in `ix`.
    std::array<double, NUM_SFB_LONG> nsa_band{};
    std::array<bool, NUM_SFB_LONG> is_ext{};
  } best{};
  bool found = false;
  /// Ranking key for `best`: perceptual cost + λ_R · real total rate
  /// (part2_3 + part2). Kept separate from `best.cum_dist`, which stays the
  /// sum of the per-band hybrid costs the downstream stats and hf_rdo expect.
  double best_obj = 0.0;

  // ---- Shaping-alpha candidates -----------------------------------------
  // α is the error-feedback coefficient in `quantize_band`
  // (`target = x − α·err`), and it is a genuine two-sided trade:
  //
  //   α → 1  the whole quantization error is pushed into the next bin, so the
  //          band keeps its level but the noise smears across it — audible as
  //          bleeding / roughness around tonal peaks.
  //   α → 0  plain rounding, so every coefficient inside the dead zone drops
  //          to nothing and the band develops spectral holes (黒点).
  //
  // Neither end is right per granule, which is exactly what an RD search is
  // for — and the search already exists: the per-band cache below is rebuilt
  // per α and every candidate is ranked by the same hybrid cost (mask-relative
  // NMR blended with waveform SDR by `m_alpha`, plus λ_R·bits), so both
  // failure modes are already priced. What was missing is candidates: the list
  // used to hold the tonality heuristic's own value plus the raw config value,
  // which differ by a few hundredths and sit on the same side of the axis, so
  // the search could never actually make the trade.
  //
  // Offer a ladder spanning the axis instead, always including the heuristic's
  // pick so the previous behaviour stays reachable, and take the points
  // *furthest* from it first — with only two probes the useful second one is
  // the opposite failure mode, not a neighbour.
  constexpr std::array<double, 4> NSA_LADDER = {0.10, 0.35, 0.60, 0.85};
  // Each extra α rebuilds the whole (sfb, sf) cache for every gain/scale/
  // preflag, so this is the most expensive axis in the search. Derive the
  // count from effort; the floor of 2 keeps the default at the cost it
  // already paid (the old list was usually 2 entries too).
  const int n_nsa = std::clamp(s.cfg_rate_probes / 2, 2, 4);

  std::vector<double> nsa_candidates;
  nsa_candidates.push_back(nsa);
  if (!nsa_disabled) {
    std::vector<double> rest(NSA_LADDER.begin(), NSA_LADDER.end());
    std::sort(rest.begin(), rest.end(), [eff_nsa](double a, double b) {
      return std::abs(a - eff_nsa) > std::abs(b - eff_nsa);
    });
    for (double v : rest) {
      if (static_cast<int>(nsa_candidates.size()) >= n_nsa)
        break;
      // On an attack or its pre-window the heuristic lowered α deliberately,
      // to stop the error feedback diffusing quantization noise into the
      // quiet run-up. The hybrid cost has no temporal term — TRANSIENT_WEIGHT
      // scales a band's noise, it does not know *when* inside the window that
      // noise lands — so a search allowed to raise α there will happily undo
      // that protection for a frequency-domain gain it can see. Measured with
      // the ladder unconstrained: pre_echo 0 better / 3 worse, mean −0.98 dB.
      // Let the search go quieter than the heuristic, never louder.
      // Nothing to search on an attack or its run-up: the heuristic's α is the
      // only value there. Both directions were measured and both hurt, for
      // reasons the cost cannot see because it has no temporal term.
      //   α above the heuristic — the error feedback diffuses noise across the
      //     window and part of it lands before the attack.
      //   α below it — plain rounding drops the quiet run-up's coefficients to
      //     zero, and an emptied run-up *is* the pre-echo measurement. This is
      //     the one that actually bit: restricting the ladder to α ≤ heuristic
      //     left attack_from_silence@64 5.5 dB worse on pre_echo, unchanged
      //     from the unrestricted ladder.
      // The temporal protection belongs to the transient gate; leave it alone
      // and take the per-band freedom everywhere else.
      if (granule_is_transient || is_pre_window || frame_has_attack)
        continue;
      bool dup = false;
      for (double have : nsa_candidates)
        dup |= std::abs(have - v) <= 0.02;
      if (!dup)
        nsa_candidates.push_back(v);
    }
  }

  // The per-band cache below — quantize_band + band_distortion + the hybrid
  // perceptual cost + the bit lower bound for every (sfb, sf) — is a pure
  // function of (nsa, scale, preflag, gain). It does NOT depend on
  // scalefac_compress, which only selects the per-band sf cap and the part2
  // length. Build it ONCE per (nsa, scale, preflag, gain) over the full
  // candidate range [0,SF_MAX_CANDIDATE] and solve via RDO.
  {
    for (int scale : SCALE_CANDIDATES) {
      for (int preflag : PREFLAG_CANDIDATES) {
        for (int gain : viable_gains) {
          const std::int64_t scalei_global = s.steptabi[gain + 127];
          const double steptab_global = s.steptab[gain + 127];

          const auto t_cache = prof_clock::now();
          // Reset + build the full cache (sf = 0..SF_MAX_CANDIDATE).
          for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb)
            for (int sf = 0; sf <= sf_max; ++sf)
              cache[sfb][sf].ok = false;
          for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
            const double lo_hz = band_table[sfb] * s.wave.samplerate / 1152.0;
            const bool is_below_200 = (lo_hz < 200.0);
            for (int sf = 0; sf <= sf_max; ++sf) {
             for (double nsa_val : nsa_candidates) {
              auto q = quantize_band(
                  xrabs_bands[sfb], band_lens[sfb], scalei_global,
                  steptab_global, sf, scale, preflag, sfb, s.int2idx, nsa_val,
                  zb_factor[static_cast<std::size_t>(sfb)], zb_fill_exp,
                  prev_bands[sfb], hyst_scale);
              if (!q.ok)
                continue;

              // Active noise injection for noise-fill SFBs.
              // Whenever the natural quantization zeroes the
              // band, write a sparse ix=1 pattern so the
              // decoder synthesises noise at the source's own
              // band level. Fires only when the source had
              // content there — a band that was dark in the
              // source is not a hole and gets no fill.
              const double nf_w =
                  band_is_comb[static_cast<std::size_t>(sfb)]
                      ? 0.0
                      : noise_fill_weight(sfb, mt.tonality[sfb], nf_first);
              static const bool no_inject =
                  (std::getenv("RMPS_NO_INJECT") != nullptr);
              if (!no_inject && nf_w > 0.0 && q.max_val == 0 &&
                  !granule_is_transient && !is_pre_window) {
                // Seed the hole toward the source's own band level
                // (envelope-tracking) graded by nf_w. The budget is
                // the source band energy itself: the former aim of
                // max(Thr, E_band) filled *above* the source
                // whenever the threshold over-estimated, which is
                // exactly when the fill turns into audible extra
                // hiss. At or below the source level the fill is
                // indistinguishable from the content it replaces
                // while still closing the spectrogram hole.
                const double budget_e = mt.e_band[sfb] * nf_w;
                // Aim closer to the source level as the rate
                // starves (still bounded by budget_e).
                const double target_frac =
                    std::min(0.97, ((lo_hz >= 12000.0) ? 0.90 : 0.60) +
                                       0.10 * low_rate_t);
                const double target_e = budget_e * target_frac;
                const double per_bin = q.dequant_factor * q.dequant_factor;
                // A single coded bin may not exceed the band budget
                // by more than ~0.5 dB. Under a coarser gain the
                // fill would land louder than the source and read
                // as a boost — skip, and let a finer-gain candidate
                // do the filling instead. At starved rates the
                // allowance relaxes toward ~+2 dB: a slightly hot
                // fill beats a hole there.
                const bool can_inject =
                    per_bin <= budget_e * (1.12 + 0.5 * low_rate_t);
                if (per_bin > 0 && can_inject && target_e > 0) {
                  int n_target = static_cast<int>(target_e / per_bin + 0.5);
                  n_target = std::clamp(n_target, 1, band_lens[sfb]);
                  // Trim so the injected total never exceeds the
                  // source band energy.
                  while (n_target > 1 && n_target * per_bin > budget_e)
                    --n_target;
                  const int stride_inj = std::max(1, band_lens[sfb] / n_target);
                  for (int pos = stride_inj / 2; pos < band_lens[sfb];
                       pos += stride_inj) {
                    q.ix[static_cast<std::size_t>(pos)] = 1;
                  }
                }
              }

              CacheEntry e;
              e.ok = true;
              e.ix = q.ix;
              double e_dens = 1.0, e_def = -1.0, e_exc = 0.0, e_sfm = 1e9;
              e.D_raw = band_distortion(xr_bands[sfb], q.ix, q.dequant_factor,
                                        &e.E_recon, &e_dens,
                                        &e_def, &e_exc, &e_sfm);
              e.deficit = e_def;
              e.excess = e_exc;
              const BandRate br_band = band_rate(q.ix);
              e.bits_est = br_band.total;
              e.rate_cost = br_band.marginal;
              e.dist = hybrid_band_cost(
                  sfb, e.D_raw, e.E_recon, adj_thr[sfb], mt.e_band[sfb],
                  mt.tonality[sfb], adj_weights[sfb], e.rate_cost, m_alpha,
                  m_lambda_R, granule_is_transient, eff_hf_boost, is_pre_window,
                  e_ref, gr_energy_contrast, is_below_200, m_waveform_reg,
                  is_side, band_is_comb[static_cast<std::size_t>(sfb)],
                  eff_hole_surcharge, nf_first, e_dens, e_def, e_exc, e_sfm);
              e.nsa = nsa_val;

              // If online HF extension is active, evaluate synthesized HF candidate for
              // cutoff bands (sfb >= s.hf_cutoff_sfb) up to 20000 Hz.
              // If its RD cost is lower, replace the candidate for this subband.
              const bool is_cutoff_band = (sfb >= s.hf_cutoff_sfb);
              if (s.cfg_hf_extension && lo_hz <= 20000.0 && has_ext_hf[sfb] && is_cutoff_band) {
                auto q_ext = quantize_band(
                    xrabs_ext_bands[sfb], band_lens[sfb], scalei_global,
                    steptab_global, sf, scale, preflag, sfb, s.int2idx, nsa_val,
                    zb_factor[static_cast<std::size_t>(sfb)], zb_fill_exp,
                    prev_bands[sfb], hyst_scale);
                if (q_ext.ok) {
                  CacheEntry e_ext;
                  e_ext.ok = true;
                  e_ext.is_ext = true;
                  e_ext.ix = q_ext.ix;
                  double ext_dens = 1.0, ext_def = -1.0, ext_exc = 0.0, ext_sfm = 1e9;
                  e_ext.D_raw = band_distortion(xr_ext_bands[sfb], q_ext.ix, q_ext.dequant_factor,
                                                &e_ext.E_recon, &ext_dens,
                                                &ext_def, &ext_exc, &ext_sfm);
                  e_ext.deficit = ext_def;
                  e_ext.excess = ext_exc;
                  const BandRate br_ext = band_rate(q_ext.ix);
                  e_ext.bits_est = br_ext.total;
                  e_ext.rate_cost = br_ext.marginal;
                  e_ext.dist = hybrid_band_cost(
                      sfb, e_ext.D_raw, e_ext.E_recon, adj_thr[sfb], mt.e_band[sfb],
                      mt.tonality[sfb], adj_weights[sfb], e_ext.rate_cost, m_alpha,
                      m_lambda_R, granule_is_transient, eff_hf_boost, is_pre_window,
                      e_ref, gr_energy_contrast, is_below_200, m_waveform_reg,
                      is_side, band_is_comb[static_cast<std::size_t>(sfb)],
                      eff_hole_surcharge, nf_first, ext_dens, ext_def, ext_exc, ext_sfm);
                  e_ext.nsa = nsa_val;

                  bool replace = false;
                  if (is_cutoff_band) {
                    replace = (q.max_val == 0) || (e_ext.dist < e.dist);
                  } else {
                    replace = (e_ext.dist < e.dist);
                  }

                  if (replace) {
                    e = std::move(e_ext);
                  }
                }
              }
              // Keep the cheapest alpha for *this* band. `e.dist` already
              // carries λ_R·bits, so this is a proper per-band RD decision
              // and not just "least distortion".
              auto &slot = cache[static_cast<std::size_t>(sfb)]
                                [static_cast<std::size_t>(sf)];
              if (!slot.ok || e.dist < slot.dist)
                slot = std::move(e);
             }
            }
          }

          if (prof_on()) beam_prof().cache += secs(t_cache);
          const auto t_dp = prof_clock::now();
          for (int compress : COMPRESS_CANDIDATES) {
            const int slen1 =
                tables::slen1_tab[static_cast<std::size_t>(compress)];
            const int slen2 =
                tables::slen2_tab[static_cast<std::size_t>(compress)];

            std::array<int, NUM_SFB_LONG> sf_caps{};
            for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
              const int slen = (sfb < 11) ? slen1 : slen2;
              int cap = (slen > 0) ? ((1 << slen) - 1) : 0;
              cap = std::min(cap, sf_max);
              sf_caps[sfb] = cap;
            }

            for (std::size_t pi = 0; pi < scfsi_count; ++pi) {
              const auto &scfsi_pat = (gr == 1 && gr0_sf != nullptr && mpeg1)
                                          ? SCFSI_PATTERNS[pi]
                                          : SCFSI_PATTERNS[0];

              const int part2 = part2_length_for(compress, scfsi_pat, gr);
              const int huff_budget = search_bits - part2;
              if (huff_budget < 0)
                continue;

              // ---- Rate-bucketed DP over the scalefactor bands ----
              // The stage cost is additive and a function of (sfb, sf) alone,
              // so ranking partial paths by cost — what the beam did —
              // reproduces the per-band argmin exactly: every survivor it kept
              // was that same allocation with one band nudged, and all of them
              // therefore spent nearly the same number of bits. When the winner
              // overflowed the real bit count the rest overflowed with it, and
              // the granule dropped to the reset-scalefactor fallback.
              //
              // The one thing that couples the bands is the bit budget, which
              // makes this a multiple-choice knapsack. Solve it as one: keep
              // the cheapest path *per bucket of cumulative rate*, so the
              // survivors span the whole rate axis instead of clustering at a
              // point. Exact up to the bucket quantisation, same order of work
              // as the beam.
              //
              // The DP tracks rate up to `overflow_cap`, not to the budget. Its
              // axis is an estimate, and the scalefactor range alone (0..cap,
              // ~1.5 dB a step) cannot always coarsen a dense granule under the
              // budget — so a hard prune at `huff_budget` lets a stage run out
              // of states entirely and drops the granule to the fallback even
              // though a candidate a few percent over the estimate would have
              // passed the real bit count. Carrying the overflow band costs
              // nothing (those states just lose the frontier ranking below) and
              // keeps the search alive exactly where it matters.
              const int overflow_cap = huff_budget + huff_budget / 4 + 64;
              for (auto &c : dp_cur)
                c.ok = false;
              dp_cur[0] = {0.0, 0, 0, true};
              const double bucket_scale =
                  static_cast<double>(n_buckets - 1) /
                  static_cast<double>(std::max(overflow_cap, 1));
              bool feasible = true;

              for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
                const double lo_hz =
                    band_table[sfb] * s.wave.samplerate / 1152.0;
                const bool is_below_200 = (lo_hz < 200.0);
                const int sb = scfsi_band_of_sfb(sfb);
                int forced_sf = -1;
                if (gr == 1 && gr0_sf != nullptr && scfsi_pat[sb] == 1)
                  forced_sf = gr0_sf[sfb];

                const int cap = sf_caps[sfb];
                for (auto &c : dp_nxt)
                  c.ok = false;
                // Band hole guard: an empty candidate is not admissible for a
                // protected band, but only when a non-empty one exists within
                // this compress's cap — never at the price of feasibility.
                //
                // "A non-empty one exists" is not enough: it also has to be at
                // roughly the right *level*. The smallest non-empty candidate a
                // band can have is a single |ix| = 1, which reconstructs to the
                // band's `dequant_factor` — the bottom of the codebook at this
                // step — and that is unrelated to what the band actually
                // contains. For a band whose every coefficient sits under the
                // dead zone, forcing it non-empty therefore injects a level
                // *above* the source, sometimes far above.
                //
                // That is audible and it is what this guard was doing. Measured
                // on `avicii` at 64 kbps against the source's own local peak:
                // isolated samples at +11 to +17 dB, in quiet passages, at 20x
                // the rate the source itself has such peaks — a click on top of
                // the music. Turning the guard off removed the worst of them
                // (max +16.8 -> +13.3 dB, samples over +6 dB 0.020 % ->
                // 0.013 %), which is how it was attributed.
                //
                // The rescue pass further down already got this right — it
                // checks `e_rec <= t_src * e_src` before it revives a band — so
                // this is the same admissibility test, applied where the
                // constraint is imposed rather than only where it is repaired.
                // A band with no acceptable non-empty candidate is one the step
                // cannot represent at all; leaving it empty is the quieter
                // error, and the guard's own purpose (stop a *steady* band from
                // flickering to nothing) does not extend to bands that are
                // below the codeable floor to begin with.
                static const double guard_over = [] {
                  double db = 3.0;
                  if (const char *e = std::getenv("RMPS_HOLE_GUARD_OVER_DB")) {
                    const double v = std::atof(e);
                    if (v >= 0.0 && v <= 60.0) db = v;
                  }
                  return std::pow(10.0, db / 10.0);   // energy ratio
                }();
                bool skip_empty = false;
                if (hole_guard[static_cast<std::size_t>(sfb)] &&
                    forced_sf < 0 && !is_below_200) {
                  const auto &row = cache[static_cast<std::size_t>(sfb)];
                  const double e_src = mt.e_band[static_cast<std::size_t>(sfb)];
                  for (int t = 0;
                       t <= cap && t < static_cast<int>(row.size()); ++t)
                    if (row[static_cast<std::size_t>(t)].ok &&
                        row[static_cast<std::size_t>(t)].E_recon > 0.0 &&
                        row[static_cast<std::size_t>(t)].E_recon <=
                            guard_over * e_src) {
                      skip_empty = true;
                      break;
                    }
                }

                for (int r = 0; r < n_buckets; ++r) {
                  if (!dp_cur[static_cast<std::size_t>(r)].ok)
                    continue;
                  const auto &from = dp_cur[static_cast<std::size_t>(r)];
                  auto try_sf = [&](int sf_try) {
                    if (sf_try < 0)
                      return;
                    if (sf_try > cap && forced_sf < 0)
                      return;
                    auto &entry = cache[static_cast<std::size_t>(sfb)]
                                       [static_cast<std::size_t>(sf_try)];
                    if (!entry.ok)
                      return;
                    if (skip_empty && entry.E_recon <= 0.0)
                      return;
                    const int nb = from.bits + entry.bits_est;
                    if (nb > overflow_cap)
                      return;
                    const double nc = from.cost + entry.dist;
                    const auto rb = static_cast<std::size_t>(
                        static_cast<double>(nb) * bucket_scale);
                    auto &cell = dp_nxt[rb];
                    if (cell.ok && cell.cost <= nc)
                      return;
                    cell = {nc, nb, from.mbits + entry.rate_cost, true};
                    dp_sf[static_cast<std::size_t>(sfb)][rb] =
                        static_cast<std::int8_t>(sf_try);
                    dp_prev[static_cast<std::size_t>(sfb)][rb] =
                        static_cast<std::int16_t>(r);
                  };
                  if (forced_sf >= 0) {
                    // The cache spans the full [0,SF_MAX_CANDIDATE] range
                    // irrespective of compress, so gate the forced
                    // (scfsi-shared) sf by this compress's cap.
                    if (forced_sf <= cap &&
                        forced_sf <
                            static_cast<int>(
                                cache[static_cast<std::size_t>(sfb)].size()) &&
                        cache[static_cast<std::size_t>(sfb)]
                             [static_cast<std::size_t>(forced_sf)]
                                 .ok)
                      try_sf(forced_sf);
                  } else if (is_below_200) {
                    try_sf(cap);
                  } else {
                    for (int sf_try = 0; sf_try <= cap; ++sf_try)
                      try_sf(sf_try);
                  }
                }

                bool any = false;
                for (const auto &c : dp_nxt)
                  if (c.ok) {
                    any = true;
                    break;
                  }
                if (!any) {
                  feasible = false;
                  break;
                }
                dp_cur = dp_nxt;
              }
              if (!feasible)
                continue;

              // ---- Walk the rate frontier against the real bit count ----
              // The DP's rate axis is an estimate (within ~5% of the coded
              // size), so the state sitting nearest the budget may still
              // overflow. Probe the terminal states cheapest-first and keep the
              // best one that genuinely fits — the diversity the beam never
              // had.
              std::array<int, NUM_RATE_BUCKETS> order{};
              int n_order = 0;
              for (int r = 0; r < n_buckets; ++r) {
                const auto &c = dp_cur[static_cast<std::size_t>(r)];
                if (c.ok && c.bits <= huff_budget)
                  order[static_cast<std::size_t>(n_order++)] = r;
              }
              if (n_order > 0) {
                // Normal case: rank the in-budget states by cost, breaking ties
                // toward the state that spends more. Ties are not exotic here —
                // the NMR hinge scores every sub-threshold band at exactly
                // zero, so a tonal granule that masks itself well produces a
                // whole run of zero-cost states at different rates, and the
                // λ_R·bits term is far too small to separate them. Without the
                // tie-break the winner is whichever bucket `std::sort` happened
                // to leave first, i.e. the encode quietly throws away bits it
                // was granted.
                std::sort(
                    order.begin(), order.begin() + n_order, [&](int a, int b) {
                      const auto &ca = dp_cur[static_cast<std::size_t>(a)];
                      const auto &cb = dp_cur[static_cast<std::size_t>(b)];
                      if (ca.cost != cb.cost)
                        return ca.cost < cb.cost;
                      return ca.bits > cb.bits;
                    });
              } else {
                // Nothing fits the estimate. Rather than surrender the granule
                // to the fallback, offer the leanest over-budget states to the
                // real bit count — the estimate runs a few percent high on
                // dense granules, so these frequently do fit.
                for (int r = 0; r < n_buckets; ++r)
                  if (dp_cur[static_cast<std::size_t>(r)].ok)
                    order[static_cast<std::size_t>(n_order++)] = r;
                std::sort(order.begin(), order.begin() + n_order,
                          [&](int a, int b) {
                            return dp_cur[static_cast<std::size_t>(a)].bits <
                                   dp_cur[static_cast<std::size_t>(b)].bits;
                          });
              }

              std::array<int, GRANULE_SAMPLES> tmp_ix{};
              std::array<std::int8_t, NUM_SFB_LONG> sf_path{};
              for (int k = 0; k < n_order && k < s.cfg_rate_probes; ++k) {
                const auto &st = dp_cur[static_cast<std::size_t>(
                    order[static_cast<std::size_t>(k)])];
                if (found &&
                    st.cost + m_lambda_R * static_cast<double>(part2) >=
                        best_obj)
                  break;
                int r = order[static_cast<std::size_t>(k)];
                for (int sfb = NUM_SFB_LONG - 1; sfb >= 0; --sfb) {
                  sf_path[static_cast<std::size_t>(sfb)] =
                      dp_sf[static_cast<std::size_t>(sfb)]
                           [static_cast<std::size_t>(r)];
                  r = dp_prev[static_cast<std::size_t>(sfb)]
                             [static_cast<std::size_t>(r)];
                }
                materialize_ix(sf_path, cache, band_table, tmp_ix.data());
                GrInfo tmp_gi{};
                const int bits = final_bits(tmp_ix.data(), tmp_gi, s);
                if (std::getenv("RMPS_EST_DEBUG"))
                  std::fprintf(stderr,
                               "[est] f=%lld gr=%d ch=%d est=%d real=%d "
                               "ratio=%.4f budget=%d %s\n",
                               s.dbg_frame_no, gr, ch, st.bits, bits,
                               static_cast<double>(bits) /
                                   std::max(st.bits, 1),
                               huff_budget,
                               (bits > huff_budget) ? "OVER" : "fits");
                if (bits > huff_budget)
                  continue;
                // Compare candidates — across gain/scale/preflag/compress/scfsi
                // too — on perceptual cost + λ_R · *real* total rate, part2
                // included. The old comparison used the estimated rate and left
                // part2 out entirely, which quietly favoured the
                // scalefac_compress / scfsi choices that spend the most side
                // info: their extra bits never appeared in the score.
                const double obj =
                    st.cost +
                    m_lambda_R * static_cast<double>(bits + part2 - st.mbits);
                if (found && obj >= best_obj)
                  continue;
                best.sf = sf_path;
                best.ix = tmp_ix;
                best.cum_dist = st.cost;
                best.dist_band.resize(NUM_SFB_LONG);
                best.raw_band.resize(NUM_SFB_LONG);
                best.def_band.resize(NUM_SFB_LONG);
                best.exc_band.resize(NUM_SFB_LONG);
                for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
                  const auto &ce =
                      cache[static_cast<std::size_t>(sfb)]
                           [static_cast<std::size_t>(
                               sf_path[static_cast<std::size_t>(sfb)])];
                  best.dist_band[static_cast<std::size_t>(sfb)] = ce.dist;
                  best.raw_band[static_cast<std::size_t>(sfb)] = ce.D_raw;
                  best.def_band[static_cast<std::size_t>(sfb)] = ce.deficit;
                  best.exc_band[static_cast<std::size_t>(sfb)] = ce.excess;
                  best.is_ext[static_cast<std::size_t>(sfb)] = ce.is_ext;
                }
                best.tmp_gi = tmp_gi;
                best.bits = bits;
                best.part2_length = part2;
                best.compress = compress;
                for (int b = 0; b < NUM_SFB_LONG; ++b)
                  best.nsa_band[static_cast<std::size_t>(b)] =
                      cache[static_cast<std::size_t>(b)]
                           [static_cast<std::size_t>(
                                sf_path[static_cast<std::size_t>(b)])]
                               .nsa;
                best.scale = scale;
                best.preflag = preflag;
                best.scfsi_pat = scfsi_pat;
                best.gain = gain;
                best_obj = obj;
                found = true;
              }
            }
          }
          if (prof_on()) beam_prof().dp += secs(t_dp);
        }
      }
    }
  }

    if (!found) {
      if (std::getenv("RMPS_RATE_DEBUG"))
        std::fprintf(stderr,
                     "[bs] fallback f=%lld gr=%d ch=%d budget=%d search=%d "
                     "seed=%d xrmax=%d\n",
                     s.dbg_frame_no, gr, ch, max_bits, search_bits, seed_gain,
                     s.xrmax);
      // Guaranteed-feasible in-model fallback. The beam keeps the lowest-
      // distortion (= highest-bit) survivors, so when the budget is tight
      // every survivor can overflow huff_budget on the real bit count and
      // the search yields nothing — previously handing off to the legacy
      // outer_loop, whose reset scalefactors cause audible LF
      // discontinuities ("boff"). Instead we escalate the global gain,
      // using the *same* shaping quantizer the trellis uses, until the
      // all-sf=0 encoding actually fits, and build `best` from it. The
      // greedy ±1 polish below then refines the per-band sf allocation
      // against the masking thresholds within budget. At the coarse
      // extreme all coefficients quantize to zero, so this terminates.
      const int compress = COMPRESS_CANDIDATES[0];
      const int scale = 0;
      // `preflag` is searched here, not fixed at 0. This path exists because
      // nothing fit the budget, and preflag is the one control that buys
      // coarseness *for free and only where it is cheapest to give it up*:
      // it adds pretab (0 below sfb 11, rising to 3 around 13-16 kHz) to the
      // effective scalefactor, costs no bits — the flag is in the side info
      // either way — and touches nothing below sfb 11. The alternative this
      // loop had, escalating the global gain, coarsens the bass and mids by
      // the same step as the top. Taking preflag first lets a *finer* gain
      // fit, so the starved granule gives up its 13 kHz region instead of its
      // fundamentals. The fallback is not a rare path — it fired 450 times at
      // 64 kbps and 572 at 320 across four testset clips.
      int preflag = 0;
      const auto &scfsi_pat = SCFSI_PATTERNS[0];
      const int part2 = part2_length_for(compress, scfsi_pat, gr);
      const int huff_budget = std::max(0, search_bits - part2);

      // Gain index is steptab[gain+127] with the table sized [0,255], so the
      // valid gain range is [-127, 127]; larger gain ⇒ smaller scalei ⇒
      // coarser ⇒ fewer bits, with gain=127 quantizing ~everything to zero
      // (the guaranteed-feasible extreme). Escalate from the bin-search seed
      // up to 127 and take the *finest* gain that fits. NOTE: feasibility
      // is tracked with an explicit flag, not a negative sentinel.
      int chosen_gain = 0;
      bool fb_found = false;
      std::array<int, GRANULE_SAMPLES> fb_ix{};
      GrInfo fb_gi{};
      int fb_bits = 0;
      const int gstart = std::min(127, std::max(seed_gain, -127));
      // Gain is the outer loop and preflag the inner one, so the first fit
      // found is the one with the finest gain — preflag is preferred to a
      // coarser global step, which is the whole point of trying it here.
      for (int g = gstart; g <= 127 && !fb_found; ++g) {
       for (int pf : PREFLAG_CANDIDATES) {
        const std::int64_t scalei_g = s.steptabi[g + 127];
        const double steptab_g = s.steptab[g + 127];
        std::array<int, GRANULE_SAMPLES> t{};
        std::fill(t.begin(), t.end(), 0);
        bool band_ok = true;
        for (int sfb = 0; sfb < NUM_SFB_LONG && band_ok; ++sfb) {
          const double lo_hz = band_table[sfb] * s.wave.samplerate / 1152.0;
          const bool is_below_200 = (lo_hz < 200.0);
          int sf_val = 0;
          if (is_below_200) {
            const int slen =
                (sfb < 11)
                    ? tables::slen1_tab[static_cast<std::size_t>(compress)]
                    : tables::slen2_tab[static_cast<std::size_t>(compress)];
            int cap = (slen > 0) ? ((1 << slen) - 1) : 0;
            sf_val = std::min(cap, sf_max);
          }
          const bool is_ext_band = (s.cfg_hf_extension && lo_hz <= 20000.0 &&
                                    has_ext_hf[sfb] && sfb >= s.hf_cutoff_sfb);
          const std::int32_t *xrabs_ptr =
              is_ext_band ? xrabs_ext_bands[sfb] : xrabs_bands[sfb];
          auto q = quantize_band(
              xrabs_ptr, band_lens[sfb], scalei_g, steptab_g, sf_val,
              scale, pf, sfb, s.int2idx, nsa,
              zb_factor[static_cast<std::size_t>(sfb)], zb_fill_exp,
              prev_bands[sfb], hyst_scale);
          if (!q.ok) {
            band_ok = false;
            break;
          }
          const int start = band_table[sfb];
          for (int j = 0; j < band_lens[sfb]; ++j)
            t[static_cast<std::size_t>(start + j)] =
                q.ix[static_cast<std::size_t>(j)];
        }
        if (!band_ok)
          continue;
        GrInfo tg{};
        const int bits = final_bits(t.data(), tg, s);
        if (bits <= huff_budget) {
          chosen_gain = g;
          preflag = pf;
          fb_ix = t;
          fb_gi = tg;
          fb_bits = bits;
          fb_found = true;
          break;
        }
       }
      }
      if (!fb_found)
        return out; // pathological: part2 alone > budget

      const std::int64_t scalei_global = s.steptabi[chosen_gain + 127];
      const double steptab_global = s.steptab[chosen_gain + 127];
      best.sf.fill(0);
      best.ix = fb_ix;
      best.is_ext.fill(false);
      best.dist_band.assign(NUM_SFB_LONG, 0.0);
      best.raw_band.assign(NUM_SFB_LONG, 0.0);
      best.def_band.assign(NUM_SFB_LONG, 0.0);
      best.exc_band.assign(NUM_SFB_LONG, 0.0);
      double cum = 0.0;
      for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const double lo_hz = band_table[sfb] * s.wave.samplerate / 1152.0;
        const bool is_below_200 = (lo_hz < 200.0);
        int sf_val = 0;
        if (is_below_200) {
          const int slen =
              (sfb < 11)
                  ? tables::slen1_tab[static_cast<std::size_t>(compress)]
                  : tables::slen2_tab[static_cast<std::size_t>(compress)];
          int cap = (slen > 0) ? ((1 << slen) - 1) : 0;
          sf_val = std::min(cap, sf_max);
        }
        best.sf[static_cast<std::size_t>(sfb)] =
            static_cast<std::int8_t>(sf_val);

        const bool is_ext_band = (s.cfg_hf_extension && lo_hz <= 20000.0 &&
                                  has_ext_hf[sfb] && sfb >= s.hf_cutoff_sfb);
        best.is_ext[static_cast<std::size_t>(sfb)] = is_ext_band;
        const std::int32_t *xrabs_ptr =
            is_ext_band ? xrabs_ext_bands[sfb] : xrabs_bands[sfb];
        const std::int32_t *xr_ptr =
            is_ext_band ? xr_ext_bands[sfb] : xr_bands[sfb];

        auto q = quantize_band(
            xrabs_ptr, band_lens[sfb], scalei_global, steptab_global,
            sf_val, scale, preflag, sfb, s.int2idx, nsa,
            zb_factor[static_cast<std::size_t>(sfb)], zb_fill_exp,
            prev_bands[sfb], hyst_scale);
        double E_recon = 0.0;
        double b_dens = 1.0, b_def = -1.0, b_exc = 0.0, b_sfm = 1e9;
        const double D_raw =
            q.ok ? band_distortion(xr_ptr, q.ix, q.dequant_factor,
                                   &E_recon, &b_dens,
                                   &b_def, &b_exc, &b_sfm)
                 : 0.0;
        const int bits_est = q.ok ? band_rate(q.ix).marginal : 0;
        const double dist = hybrid_band_cost(
            sfb, D_raw, E_recon, adj_thr[sfb], mt.e_band[sfb], mt.tonality[sfb],
            adj_weights[sfb], bits_est, m_alpha, m_lambda_R,
            granule_is_transient, eff_hf_boost, is_pre_window, e_ref,
            gr_energy_contrast, is_below_200, m_waveform_reg, is_side,
            band_is_comb[static_cast<std::size_t>(sfb)], eff_hole_surcharge,
            nf_first, b_dens, b_def, b_exc, b_sfm);
        static const bool dens_debug =
            (std::getenv("RMPS_DENS_DEBUG") != nullptr);
        if (dens_debug && sfb >= 11)
          std::fprintf(stderr,
                       "[dens] sfb=%d dens=%.3f ton=%.3f over=%.3f "
                       "wd=%.3f wo=%.3f\n",
                       sfb, b_dens, mt.tonality[sfb],
                       E_recon / std::max(mt.e_band[sfb], 1.0),
                       nf_density_weight(b_dens),
                       nf_overshoot_weight(E_recon, mt.e_band[sfb]));
        best.dist_band[static_cast<std::size_t>(sfb)] = dist;
        best.raw_band[static_cast<std::size_t>(sfb)] = D_raw;
        best.def_band[static_cast<std::size_t>(sfb)] = b_def;
        best.exc_band[static_cast<std::size_t>(sfb)] = b_exc;
        cum += dist;
      }
      best.cum_dist = cum;
      best.tmp_gi = fb_gi;
      best.bits = fb_bits;
      best.part2_length = part2;
      best.compress = compress;
      best.nsa_band.fill(nsa);
      best.scale = scale;
      best.preflag = preflag;
      best.scfsi_pat = scfsi_pat;
      best.gain = chosen_gain;
      found = true;
    }

    // --- Local greedy ±1 polish (operates on materialised ix) ---
    {
      auto scalei_global = s.steptabi[best.gain + 127];
      auto steptab_global = s.steptab[best.gain + 127];
      const int huff_budget = search_bits - best.part2_length;
      std::array<bool, NUM_SFB_LONG> locked{};
      if (gr == 1 && gr0_sf != nullptr) {
        for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb)
          locked[sfb] = (best.scfsi_pat[scfsi_band_of_sfb(sfb)] == 1);
      }

      for (int it = 0; it < s.cfg_polish_iters; ++it) {
        bool improved = false;
        for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
          if (locked[sfb])
            continue;
          const double lo_hz = band_table[sfb] * s.wave.samplerate / 1152.0;
          const bool is_below_200 = (lo_hz < 200.0);
          const int slen =
              (sfb < 11)
                  ? tables::slen1_tab[static_cast<std::size_t>(best.compress)]
                  : tables::slen2_tab[static_cast<std::size_t>(best.compress)];
          int cap = (slen > 0) ? ((1 << slen) - 1) : 0;
          cap = std::min(cap, sf_max);
          const int cur_sf = best.sf[static_cast<std::size_t>(sfb)];
          const double cur_nsa = best.nsa_band[static_cast<std::size_t>(sfb)];
          // dlt = 0 is not a no-op here: it re-opens the shaping choice at the
          // band's current scalefactor, which is the cheapest move available
          // (no rate change to speak of, and the hole/bleed trade is exactly
          // what the DP's single pass over alpha may have mis-called once the
          // neighbouring bands settled).
          for (int dlt : {-1, 0, +1}) {
            const int new_sf = cur_sf + dlt;
            if (new_sf < 0 || new_sf > cap)
              continue;
           for (double trial_nsa : nsa_candidates) {
            if (dlt == 0 && trial_nsa == cur_nsa)
              continue;
            const bool is_ext_band =
                s.cfg_hf_extension && (lo_hz <= 20000.0) && has_ext_hf[sfb] &&
                (best.is_ext[static_cast<std::size_t>(sfb)] ||
                 sfb >= s.hf_cutoff_sfb);
            const std::int32_t *xrabs_ptr =
                is_ext_band ? xrabs_ext_bands[sfb] : xrabs_bands[sfb];
            const std::int32_t *xr_ptr =
                is_ext_band ? xr_ext_bands[sfb] : xr_bands[sfb];
            auto q = quantize_band(
                xrabs_ptr, band_lens[sfb], scalei_global, steptab_global,
                new_sf, best.scale, best.preflag, sfb, s.int2idx, trial_nsa,
                zb_factor[static_cast<std::size_t>(sfb)], zb_fill_exp,
                prev_bands[sfb], hyst_scale);
            if (!q.ok)
              continue;
            if (is_ext_band && q.max_val == 0)
              continue;
            // The polish may not open a hole the DP was barred from — unless
            // what it is being made to keep instead is itself an overshoot.
            //
            // The guard exists to stop a steady band flickering to nothing. It
            // was also, unintentionally, forcing bands to keep content *above*
            // the source: this restriction alone accounts for the encoder's
            // worst time-domain spikes. Measured on `avicii` at 64 kbps against
            // the source's own local peak — bypassing only this line reproduces
            // turning the whole guard off, max +16.8 -> +13.3 dB and samples
            // over +6 dB 0.020 % -> 0.014 %, while `RMPS_HOLE_RESCUE=0` changes
            // neither. Those spikes are isolated clicks in quiet passages, at
            // 20x the rate the source itself has such peaks.
            //
            // **Off by default.** It is the DP's `skip_empty` that implements
            // the guard; this was a second copy of the rule applied after the
            // fact, and it is the one that showed up as an artefact.
            //
            // Gating it on the band's *energy* overshoot was tried and does not
            // fire: `exc_band` never exceeds `e_band` for these bands, because
            // the band energy is right. What is wrong is where that energy sits
            // in time — a band that keeps its total but concentrates it into one
            // MDCT line rings across the whole window, and its time-domain peak
            // has nothing to do with the transient shape the source had there.
            // So there is no band-domain test that separates the two cases.
            //
            // Measured, all on one binary. Cost of removing it: nothing that can
            // be found. The harness is 0 better / 0 worse on every metric at
            // every bitrate from 48 to 320; the guard's *own* target metric,
            // isolated band holes on real material (`RMPS_HOLE_ISO`), is
            // unchanged on `avicii` (1.29 % both ways) and slightly better on
            // `witchqueen` (6.76 % -> 6.71 %). Benefit: the encoder's worst
            // time-domain spikes. On `avicii` at 64 kbps, samples more than
            // 6 dB over the source's own local peak 0.020 % -> 0.014 %, worst
            // case +16.8 -> +13.3 dB.
            //
            // RMPS_HG_POLISH=1 restores it.
            static const bool hg_polish = [] {
              const char *e = std::getenv("RMPS_HG_POLISH");
              return e && std::atoi(e) != 0;
            }();
            if (hg_polish && hole_guard[static_cast<std::size_t>(sfb)] &&
                q.max_val == 0)
              continue;
            double new_E_recon = 0.0;
            double n_dens = 1.0, n_def = -1.0, n_exc = 0.0, n_sfm = 1e9;
            const double new_D_raw = band_distortion(
                xr_ptr, q.ix, q.dequant_factor, &new_E_recon, &n_dens,
                &n_def, &n_exc, &n_sfm);
            const int new_bits_est = band_rate(q.ix).marginal;
            const double new_dist = hybrid_band_cost(
                sfb, new_D_raw, new_E_recon, adj_thr[sfb], mt.e_band[sfb],
                mt.tonality[sfb], adj_weights[sfb], new_bits_est, m_alpha,
                m_lambda_R, granule_is_transient, eff_hf_boost, is_pre_window,
                e_ref, gr_energy_contrast, is_below_200, m_waveform_reg,
                is_side, band_is_comb[static_cast<std::size_t>(sfb)],
                eff_hole_surcharge, nf_first, n_dens, n_def, n_exc, n_sfm);
            if (new_dist >= best.dist_band[static_cast<std::size_t>(sfb)])
              continue;

            // Apply tentatively.
            const int start = band_table[sfb];
            std::vector<int> saved(static_cast<std::size_t>(band_lens[sfb]));
            for (int j = 0; j < band_lens[sfb]; ++j) {
              saved[static_cast<std::size_t>(j)] = best.ix[start + j];
              best.ix[start + j] = q.ix[static_cast<std::size_t>(j)];
            }
            GrInfo tmp_gi{};
            const int trial_bits = final_bits(best.ix.data(), tmp_gi, s);
            if (trial_bits <= huff_budget) {
              best.cum_dist +=
                  (new_dist - best.dist_band[static_cast<std::size_t>(sfb)]);
              best.dist_band[static_cast<std::size_t>(sfb)] = new_dist;
              best.raw_band[static_cast<std::size_t>(sfb)] = new_D_raw;
              best.def_band[static_cast<std::size_t>(sfb)] = n_def;
              best.exc_band[static_cast<std::size_t>(sfb)] = n_exc;
              best.sf[static_cast<std::size_t>(sfb)] =
                  static_cast<std::int8_t>(new_sf);
              best.nsa_band[static_cast<std::size_t>(sfb)] = trial_nsa;
              if (is_ext_band)
                best.is_ext[static_cast<std::size_t>(sfb)] = true;
              best.tmp_gi = tmp_gi;
              best.bits = trial_bits;
              improved = true;
            } else {
              for (int j = 0; j < band_lens[sfb]; ++j)
                best.ix[start + j] = saved[static_cast<std::size_t>(j)];
            }
           }
          }
        }
        if (!improved)
          break;
      }
    }

    // --- HF rate-distortion polish ---
    // Any SFB in noise-fill mode uses level-matching cost; zeroing those
    // bands via hf_rdo would punish the level match. Stop hf_rdo at the
    // first noise-fill SFB whose tonality is below the threshold.
    // Scan from the band the fill *used* to start at, not from wherever it
    // starts now.
    //
    // These two were coupled through one constant, and the coupling is much
    // bigger than it looks: `hf_rdo` zeroes bands to save bits and stops at the
    // first fill band, so raising the fill's onset hands it every band in
    // between. Moving the fill 13 -> 16 therefore also released sfb 13-15
    // (3.4-6.2 kHz) for zeroing, and that — not the fill — is what the harness
    // measured: the losses were entirely on the *tonal* sources at *high* rate
    // (two_tone_polyphony segSNR_p05 -1.05 at 96, harmonic_tone -0.71 at 96),
    // which is where real harmonics live in those bands, and there were none at
    // the low rates the fill move was aimed at.
    //
    // The stop rule only needs a band above which noise substitution is the
    // expected treatment; that is a property of the spectrum, not of wherever
    // the fill's onset has been set to.
    int hf_rdo_end_sfb = NUM_SFB_LONG;
    for (int b = HF_RDO_SCAN_FIRST; b < NUM_SFB_LONG; ++b) {
      if (mt.tonality[b] < TOP_SFB_NOISE_TONALITY_MAX) {
        hf_rdo_end_sfb = b;
        break;
      }
    }
    if (s.cfg_hf_extension && s.hf_cutoff_sfb < hf_rdo_end_sfb) {
      hf_rdo_end_sfb = s.hf_cutoff_sfb;
    }
    // Which bands were already empty before hf_rdo ran (RMPS_HOLE_ISO), so a
    // hole can be attributed to the quantizer or to hf_rdo's thinning.
    std::array<bool, NUM_SFB_LONG> empty_pre_rdo{};
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
      bool z = true;
      for (int i = band_table[sfb]; i < band_table[sfb + 1]; ++i)
        if (best.ix[static_cast<std::size_t>(i)] != 0) { z = false; break; }
      empty_pre_rdo[static_cast<std::size_t>(sfb)] = z;
    }
    const int hf_saved =
        hf_rdo(best.ix.data(), best.bits, best.dist_band, best.raw_band,
               search_bits - best.part2_length, best.gain, best.scale,
               best.preflag, xr_bands, xrabs_bands, band_lens, band_table,
               adj_weights, mt.tonality, s, best.tmp_gi,
               /*hf_start_sfb=*/11,
               /*hf_end_sfb=*/hf_rdo_end_sfb);
    out.hf_rdo_bits_saved = hf_saved;

    // --- Band hole rescue ---
    //
    // The guard can only prefer a non-empty candidate that exists. Measured
    // (RMPS_GUARD_DIAG), when a guarded band still ends up empty it is 98-100 %
    // of the time below the representable floor: at this granule's global gain
    // even sf 0 quantizes its strongest line to zero, so no candidate is
    // non-empty and the guard correctly stands down. Those are what is left of
    // the isolated holes in Mid above 6 kHz at 48 kbps.
    //
    // So write the minimal non-empty version directly: k = round(E_src/step^2)
    // lines at |ix| = 1 on the band's strongest lines, at sf 0 (the finest step
    // available; the value is free, part2 length depends only on compress), or
    // at the shared sf when scfsi locks the band.
    //
    // What it may cost in level is the whole question, because the other
    // failure the user has reported is sizzle - added HF energy. Two ways in:
    //   * within +3 dB of the *source* band energy: an ordinary quantization
    //     outcome, always admissible;
    //   * within +1.5 dB of the band's *reconstruction last granule*: the band
    //     was coded a granule ago (that is what put it under guard), so this
    //     continues the level the listener has just heard rather than adding a
    //     new one. Capped at +6 dB over the source regardless, so a band cannot
    //     be carried indefinitely at a level the source never had.
    // And it must fit the budget hf_rdo used (`search_bits - part2_length`), so
    // the sfb21 splice keeps its reserve.
    //
    // Placed after hf_rdo so hf_rdo cannot thin it away, and before the hole
    // probes below so they measure the result. RMPS_HOLE_RESCUE=0 disables;
    // RMPS_RESCUE_SRC_DB / _PREV_DB / _CAP_DB move the three limits.
    {
      static const bool rescue_on = [] {
        const char *e = std::getenv("RMPS_HOLE_RESCUE");
        return !(e && std::atoi(e) == 0);
      }();
      const auto env_db = [](const char *name, double dflt) {
        if (const char *e = std::getenv(name)) {
          const double v = std::atof(e);
          if (v >= 0.0 && v < 40.0) return v;
        }
        return dflt;
      };
      static const double t_src =
          std::pow(10.0, env_db("RMPS_RESCUE_SRC_DB", 3.0) / 10.0);
      static const double t_prev =
          std::pow(10.0, env_db("RMPS_RESCUE_PREV_DB", 1.5) / 10.0);
      static const double t_cap =
          std::pow(10.0, env_db("RMPS_RESCUE_CAP_DB", 6.0) / 10.0);
      static long long r_try = 0, r_src = 0, r_prev = 0, r_fit = 0, r_rej = 0;
      const std::int64_t sg = s.steptabi[best.gain + 127];
      const auto band_dq = [&](int sfb, int sf_val) {
        const int eff =
            sf_val + (best.preflag ? PRETAB[static_cast<std::size_t>(sfb)] : 0);
        const std::size_t si = static_cast<std::size_t>((1 + best.scale) * eff);
        if (si >= std::size(tables::pow2_half_tab)) return 0.0;
        const double sb = static_cast<double>(sg) * tables::pow2_half_tab[si];
        return sb > 0.0 ? 4294967296.0 / sb : 0.0;
      };
      if (rescue_on) {
        std::vector<int> order;
        for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
          if (!hole_guard[static_cast<std::size_t>(sfb)]) continue;
          bool z = true;
          for (int i = band_table[sfb]; i < band_table[sfb + 1]; ++i)
            if (best.ix[static_cast<std::size_t>(i)] != 0) { z = false; break; }
          if (z) order.push_back(sfb);
        }
        std::sort(order.begin(), order.end(), [&](int x, int y) {
          return mt.e_band[static_cast<std::size_t>(x)] >
                 mt.e_band[static_cast<std::size_t>(y)];
        });
        bool changed = false;
        for (int sfb : order) {
          const auto k = static_cast<std::size_t>(sfb);
          ++r_try;
          const bool locked = gr == 1 && gr0_sf != nullptr &&
                              best.scfsi_pat[scfsi_band_of_sfb(sfb)] == 1;
          const int sf_use = locked ? static_cast<int>(best.sf[k]) : 0;
          const double dq = band_dq(sfb, sf_use);
          const double e_src = mt.e_band[k];
          if (dq <= 0.0 || e_src <= 0.0) { ++r_rej; continue; }
          const double unit = dq * dq;
          const int len = band_lens[sfb];
          const int kk =
              std::clamp(static_cast<int>(e_src / unit + 0.5), 1, len);
          const double e_rec = kk * unit;
          const bool ok_src = e_rec <= t_src * e_src;
          const bool ok_prev = !ok_src && prev_erec[k] > 0.0 &&
                               e_rec <= t_prev * prev_erec[k] &&
                               e_rec <= t_cap * e_src;
          if (!ok_src && !ok_prev) { ++r_rej; continue; }
          const int start = band_table[sfb];
          std::vector<std::pair<std::int32_t, int>> lines;
          lines.reserve(static_cast<std::size_t>(len));
          for (int j = 0; j < len; ++j)
            lines.emplace_back(s.xrabs[static_cast<std::size_t>(start + j)], j);
          std::partial_sort(lines.begin(), lines.begin() + kk, lines.end(),
                            [](const auto &x, const auto &y) {
                              return x.first > y.first;
                            });
          for (int t = 0; t < kk; ++t)
            best.ix[static_cast<std::size_t>(
                start + lines[static_cast<std::size_t>(t)].second)] = 1;
          GrInfo rgi{};
          const int rbits = final_bits(best.ix.data(), rgi, s);
          if (rbits > search_bits - best.part2_length) {
            for (int t = 0; t < kk; ++t)
              best.ix[static_cast<std::size_t>(
                  start + lines[static_cast<std::size_t>(t)].second)] = 0;
            ++r_rej;
            continue;
          }
          best.bits = rbits;
          best.tmp_gi = rgi;
          best.sf[k] = static_cast<std::int8_t>(sf_use);
          BandIx slice;
          slice.size = len;
          for (int j = 0; j < len; ++j)
            slice[static_cast<std::size_t>(j)] =
                best.ix[static_cast<std::size_t>(start + j)];
          double er = 0.0, dn = 1.0, df = -1.0, ex = 0.0, sm = 1e9;
          const double d_raw =
              band_distortion(xr_bands[sfb], slice, dq, &er, &dn, &df, &ex, &sm);
          const bool is_below_200 =
              band_table[sfb] * s.wave.samplerate / 1152.0 < 200.0;
          best.raw_band[k] = d_raw;
          best.def_band[k] = df;
          best.exc_band[k] = ex;
          best.dist_band[k] = hybrid_band_cost(
              sfb, d_raw, er, adj_thr[sfb], mt.e_band[sfb], mt.tonality[sfb],
              adj_weights[sfb], band_rate(slice).marginal, m_alpha, m_lambda_R,
              granule_is_transient, eff_hf_boost, is_pre_window, e_ref,
              gr_energy_contrast, is_below_200, m_waveform_reg, is_side,
              band_is_comb[k], eff_hole_surcharge, nf_first, dn, df, ex, sm);
          changed = true;
          ++r_fit;
          if (ok_src) ++r_src; else ++r_prev;
        }
        if (changed) {
          best.cum_dist = 0.0;
          for (double d : best.dist_band)
            best.cum_dist += d;
        }
      }
      // Reconstructed band energies: the next granule's continuity reference.
      for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const auto k = static_cast<std::size_t>(sfb);
        const double dq = band_dq(sfb, static_cast<int>(best.sf[k]));
        double er = 0.0;
        for (int i = band_table[sfb]; i < band_table[sfb + 1]; ++i) {
          const double r =
              tables::pow43_tab[best.ix[static_cast<std::size_t>(i)]] * dq;
          er += r * r;
        }
        s.prev_erec_band[static_cast<std::size_t>(ch)][k] = er;
      }
      // Steady-state coding gain per band, for the hit matching in
      // iteration_loop: power-weighted, so it measures what a decoded
      // spectrogram shows (holes count as the energy they lose).
      if (!s.hit_window[gr] && s.block_type[ch][gr] == BLOCK_LONG) {
        constexpr double HIT_MATCH_ALPHA = 0.05;   // ~20 granules
        for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
          const auto k = static_cast<std::size_t>(sfb);
          const auto c = static_cast<std::size_t>(ch);
          const double es = mt.e_band[k];
          if (es <= 0.0)
            continue;
          s.hf_steady_er[c][k] +=
              HIT_MATCH_ALPHA * (s.prev_erec_band[c][k] - s.hf_steady_er[c][k]);
          s.hf_steady_es[c][k] +=
              HIT_MATCH_ALPHA * (es - s.hf_steady_es[c][k]);
        }
      }
      if (std::getenv("RMPS_GUARD_DIAG")) {
        static long long tick = 0;
        if ((++tick % 256) == 0)
          std::fprintf(stderr,
                       "[rescue] tried=%lld written=%lld (src %lld, prev %lld) "
                       "refused=%lld\n",
                       r_try, r_fit, r_src, r_prev, r_rej);
      }
    }

    // Isolated vs continuing band holes (RMPS_HOLE_ISO=1).
    //
    // A hole that persists in the same band across granules reads as a notch —
    // a spectral shape — and is clean in MDCT terms: both overlap neighbours
    // are zero too, so there is nothing for TDAC to cancel. A hole in *one*
    // granule, with the band coded on both sides, is the 黒点: the overlap-add
    // loses one half of a cancelling pair and leaks aliasing on top of the lost
    // energy. So the harm is not "a hole", it is a hole that disagrees with its
    // temporal neighbours, and this counts the two separately.
    //
    // Hole = band active in the source (within 50 dB of the granule's loudest
    // band, the same 1e-5 psyeval uses) but quantized to all zeros. Classified
    // one granule late, once the next granule's state is known. Split by
    // channel role (a Side hole is an image wobble, a Mid hole a dark spot in
    // the mix) and by whether the *source* was steady there (within 3 dB of
    // both neighbours): a steady-source isolated hole is pure decision noise.
    if (std::getenv("RMPS_HOLE_ISO")) {
      struct St {
        std::array<std::uint8_t, NUM_SFB_LONG> hole{}, act{}, rdo{};
        std::array<double, NUM_SFB_LONG> e{};
        bool valid = false;
      };
      static std::array<std::array<St, 2>, MAX_CHANNELS> h{};
      static long long n_act[4][3] = {}, n_hole[4][3] = {}, n_iso[4][3] = {},
                       n_iso_steady[4][3] = {}, n_iso_rdo[4][3] = {};
      const int role = (s.mpeg.mode_ext == 2 ? 0 : 2) + (ch & 1);
      St now;
      double emax = 0.0;
      for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb)
        emax = std::max(emax, mt.e_band[sfb]);
      for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const auto k = static_cast<std::size_t>(sfb);
        bool z = true;
        for (int i = band_table[sfb]; i < band_table[sfb + 1]; ++i)
          if (best.ix[static_cast<std::size_t>(i)] != 0) { z = false; break; }
        const bool act = emax > 0.0 && mt.e_band[sfb] > emax * 1e-5;
        now.act[k] = act;
        now.hole[k] = act && z;
        now.rdo[k] = act && z && !empty_pre_rdo[k];
        now.e[k] = mt.e_band[sfb];
      }
      now.valid = true;
      auto &hc = h[static_cast<std::size_t>(ch)];
      // The chain is only meaningful between consecutive long-layout granules.
      if (!s.prev_l3_enc_long[ch]) { hc[0].valid = hc[1].valid = false; }
      const St &pa = hc[0], &pm = hc[1];
      if (pa.valid && pm.valid) {
        const auto db = [](double x, double y) {
          return (x > 0.0 && y > 0.0) ? std::abs(10.0 * std::log10(x / y))
                                      : 99.0;
        };
        for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
          const auto k = static_cast<std::size_t>(sfb);
          const double hz = band_table[sfb] * s.wave.samplerate / 1152.0;
          const int r = hz < 2000.0 ? 0 : (hz < 6000.0 ? 1 : 2);
          if (!pm.act[k]) continue;
          ++n_act[role][r];
          if (!pm.hole[k]) continue;
          ++n_hole[role][r];
          // Isolated: coded (active and not a hole) on *both* sides.
          const bool iso = pa.act[k] && !pa.hole[k] && now.act[k] && !now.hole[k];
          if (!iso) continue;
          ++n_iso[role][r];
          if (pm.rdo[k]) ++n_iso_rdo[role][r];
          if (db(pm.e[k], pa.e[k]) < 3.0 && db(pm.e[k], now.e[k]) < 3.0)
            ++n_iso_steady[role][r];
        }
      }
      hc[0] = hc[1];
      hc[1] = now;
      static long long tick = 0;
      if ((++tick % 256) == 0) {
        static const char *rn[4] = {"M", "S", "L", "R"};
        static const char *nm[3] = {"<2k", "2-6k", "6k+"};
        for (int ro = 0; ro < 4; ++ro)
          for (int r = 0; r < 3; ++r) {
            if (!n_act[ro][r]) continue;
            std::fprintf(stderr,
                         "[holeiso] %s %-4s active=%lld holes=%.2f%% "
                         "isolated=%.2f%% (of holes %.0f%%) steady-src %.0f%% "
                         "hf_rdo %.0f%%\n",
                         rn[ro], nm[r], n_act[ro][r],
                         100.0 * n_hole[ro][r] / n_act[ro][r],
                         100.0 * n_iso[ro][r] / n_act[ro][r],
                         100.0 * n_iso[ro][r] / std::max(n_hole[ro][r], 1LL),
                         100.0 * n_iso_steady[ro][r] /
                             std::max(n_iso[ro][r], 1LL),
                         100.0 * n_iso_rdo[ro][r] / std::max(n_iso[ro][r], 1LL));
          }
      }
    }

    // Why a guarded band still ends up empty (RMPS_GUARD_DIAG=1).
    //
    // Either the guard was bypassed (the band's peak line *was* representable
    // at sf 0 under this granule's gain, so some path emptied it anyway), or
    // the band sat below the representable floor — every scalefactor quantizes
    // it to zero at this gain, and there was no non-empty candidate to prefer.
    // For the second kind, report how far over the *source* band energy a
    // single +-1 line at sf 0 would land: that is what rescuing it would cost.
    if (std::getenv("RMPS_GUARD_DIAG")) {
      static long long n_prot[4][3] = {}, n_empty[4][3] = {},
                       n_repr[4][3] = {}, n_floor[4][3] = {};
      static long long ov[4][3][5] = {};   // <=0, 0-3, 3-6, 6-10, >10 dB
      const int role = (s.mpeg.mode_ext == 2 ? 0 : 2) + (ch & 1);
      const std::int64_t sg = s.steptabi[best.gain + 127];
      for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const auto k = static_cast<std::size_t>(sfb);
        if (!hole_guard[k]) continue;
        const double hz = band_table[sfb] * s.wave.samplerate / 1152.0;
        const int r = hz < 2000.0 ? 0 : (hz < 6000.0 ? 1 : 2);
        ++n_prot[role][r];
        bool z = true;
        std::int32_t xpk = 0;
        for (int i = band_table[sfb]; i < band_table[sfb + 1]; ++i) {
          if (best.ix[static_cast<std::size_t>(i)] != 0) z = false;
          xpk = std::max(xpk, s.xrabs[static_cast<std::size_t>(i)]);
        }
        if (!z) continue;
        ++n_empty[role][r];
        const int eff_sf = best.preflag ? PRETAB[k] : 0;
        const double mult = tables::pow2_half_tab[static_cast<std::size_t>(
            (1 + best.scale) * eff_sf)];
        const std::int64_t sb =
            static_cast<std::int64_t>(static_cast<double>(sg) * mult + 0.5);
        if (sb <= 0) continue;
        const std::int64_t ln =
            (static_cast<std::int64_t>(xpk) * sb + 0x80000000LL) >> 32;
        if (ln >= 1) { ++n_repr[role][r]; continue; }
        ++n_floor[role][r];
        const double dq = 4294967296.0 / static_cast<double>(sb);
        const double e_src = std::max(mt.e_band[k], 1.0);
        const double odb = 10.0 * std::log10(dq * dq / e_src);
        const int bkt = odb <= 0.0 ? 0 : odb <= 3.0 ? 1 : odb <= 6.0 ? 2
                       : odb <= 10.0 ? 3 : 4;
        ++ov[role][r][bkt];
      }
      static long long tick = 0;
      if ((++tick % 256) == 0) {
        static const char *rn[4] = {"M", "S", "L", "R"};
        static const char *nm[3] = {"<2k", "2-6k", "6k+"};
        for (int ro = 0; ro < 4; ++ro)
          for (int r = 0; r < 3; ++r) {
            if (!n_prot[ro][r]) continue;
            const long long fl = std::max(n_floor[ro][r], 1LL);
            std::fprintf(stderr,
                "[gdiag] %s %-4s guarded=%lld empty=%.1f%% | of empty: "
                "repr %.0f%% floor %.0f%% | floor overshoot <=0:%.0f "
                "0-3:%.0f 3-6:%.0f 6-10:%.0f >10:%.0f %%\n",
                rn[ro], nm[r], n_prot[ro][r],
                100.0 * n_empty[ro][r] / n_prot[ro][r],
                100.0 * n_repr[ro][r] / std::max(n_empty[ro][r], 1LL),
                100.0 * n_floor[ro][r] / std::max(n_empty[ro][r], 1LL),
                100.0 * ov[ro][r][0] / fl, 100.0 * ov[ro][r][1] / fl,
                100.0 * ov[ro][r][2] / fl, 100.0 * ov[ro][r][3] / fl,
                100.0 * ov[ro][r][4] / fl);
          }
      }
    }

    // Where the bits actually go (RMPS_BAND_COST=1).
    //
    // Before trying to make the high band cheaper it is worth knowing what it
    // costs, and the only honest measure is marginal: Huffman coding is
    // contextual, so a band's "share" is what the granule stops paying when
    // that band is zeroed, not any per-line tally. This zeroes everything from
    // sfb X up and recounts, for a few X, and reports the saving as a fraction
    // of the granule's own bits. It also reports what an all-|ix|<=1 version of
    // the same tail would cost, which is the floor a count1-only high band
    // could reach.
    // Prize for per-band M/S attenuation (RMPS_MS_PRIZE=1).
    //
    // The proposal is a *per-band* decision, not a frequency threshold: every
    // scalefactor band independently decides how far to pull down whichever of
    // M/S is the weaker one there. Before building the search, size it — how
    // many of this granule's bits are spent on bands where this channel is the
    // dominated one, and by how much is it dominated?
    //
    // The attenuation is baked into the coefficients and never transmitted, so
    // unlike `scalefac` it costs no side info and is not tied to the band grid
    // at all. That is what makes "quite fine" bands free here.
    if (std::getenv("RMPS_MS_PRIZE") && s.mpeg.mode_ext == 2 &&
        s.wave.channels == 2) {
      // Bucket the weaker channel's bits by how far down it actually is.
      // The question the proposal turns on is not "which side is weaker" —
      // one always is — but "how often is one side so far down that pulling it
      // further costs nothing", and how many bits live there.
      static const double EDGES[5] = {6.0, 12.0, 20.0, 30.0, 1e9};
      static double bucket_bits[2][5] = {};
      static long long bucket_n[2][5] = {};
      static double all_bits[2] = {};
      static long long all_n[2] = {};
      // Split at 6 kHz: the rule of thumb this is sizing ("at high frequency
      // 9-12 dB of dominance is already enough to drop the weaker side") is
      // a statement about the top, and the binaural system stops resolving
      // interaural phase around 1.5-2 kHz anyway, so the two halves must be
      // reported apart or the low bands drown the answer.
      const double SPLIT_HZ = 6000.0;
      const auto &other = s.mdct_freq[ch ^ 1][gr];
      for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const int lo = band_table[sfb], hi = band_table[sfb + 1];
        double e_this = 0.0, e_other = 0.0;
        for (int i = lo; i < hi; ++i) {
          const double x = s.xr[i], y = other[i];
          e_this += x * x;
          e_other += y * y;
        }
        BandIx slice;
        slice.size = hi - lo;
        for (int j = 0; j < hi - lo; ++j)
          slice[static_cast<std::size_t>(j)] =
              best.ix[static_cast<std::size_t>(lo + j)];
        const double b_bits = band_rate(slice).total;
        const int half =
            (lo * s.wave.samplerate / 1152.0 >= SPLIT_HZ) ? 1 : 0;
        all_bits[half] += b_bits;
        ++all_n[half];
        if (e_this <= 0.0 || e_other <= e_this) continue;
        const double dom = 10.0 * std::log10(e_other / e_this);
        for (int k = 0; k < 5; ++k)
          if (dom < EDGES[k]) {
            bucket_bits[half][k] += b_bits;
            ++bucket_n[half][k];
            break;
          }
      }
      static long long tick = 0;
      if ((++tick % 4000) == 0) {
        static const char *nm[5] = {"  0-6 dB", " 6-12 dB", "12-20 dB",
                                    "20-30 dB", "  30+ dB"};
        for (int h = 0; h < 2; ++h) {
          std::fprintf(stderr, "[msprize] %s 6 kHz\n", h ? "ABOVE" : "BELOW");
          for (int k = 0; k < 5; ++k)
            std::fprintf(stderr,
                         "  weaker by %s : %5.1f%% of bands, %5.2f%% of bits\n",
                         nm[k],
                         100.0 * static_cast<double>(bucket_n[h][k]) /
                             static_cast<double>(all_n[h] ? all_n[h] : 1),
                         100.0 * bucket_bits[h][k] /
                             std::max(all_bits[h], 1.0));
        }
      }
    }

    if (std::getenv("RMPS_BAND_COST")) {
      static const int probes[5] = {11, 13, 15, 17, 19};
      static double frac[5] = {}, cheap[5] = {}, packed[5] = {};
      static long long n = 0;
      std::array<int, GRANULE_SAMPLES> probe{};
      GrInfo tmp{};
      std::copy_n(best.ix.data(), GRANULE_SAMPLES, probe.begin());
      const int full = final_bits(probe.data(), tmp, s);
      if (full > 0) {
        ++n;
        for (int k = 0; k < 5; ++k) {
          const int cut = band_table[probes[k]];
          std::copy_n(best.ix.data(), GRANULE_SAMPLES, probe.begin());
          std::fill(probe.begin() + cut, probe.end(), 0);
          GrInfo t2{};
          frac[k] += static_cast<double>(full - final_bits(probe.data(), t2, s)) /
                     static_cast<double>(full);
          // Same tail, clamped to the count1 alphabet.
          std::copy_n(best.ix.data(), GRANULE_SAMPLES, probe.begin());
          for (int i = cut; i < GRANULE_SAMPLES; ++i)
            probe[static_cast<std::size_t>(i)] =
                std::clamp(probe[static_cast<std::size_t>(i)], -1, 1);
          GrInfo t3{};
          cheap[k] += static_cast<double>(final_bits(probe.data(), t3, s)) /
                      static_cast<double>(full);
          // Floor for a quad-aligned high band: keep each scalefactor band's
          // nonzero *count* and signs exactly, but slide them to the front of
          // the band so they fill whole count1 quadruples instead of being
          // scattered one per quad. Band energy is preserved (same values, same
          // band), and above ~6 kHz a critical band is several sfbs wide, so
          // the ear does not resolve where inside the band they sit. This is
          // the cheapest pattern that carries the same band energy.
          std::copy_n(best.ix.data(), GRANULE_SAMPLES, probe.begin());
          for (int sfb = probes[k]; sfb < NUM_SFB_LONG; ++sfb) {
            const int lo = band_table[sfb], hi = band_table[sfb + 1];
            std::vector<int> nz;
            for (int i = lo; i < hi; ++i)
              if (probe[static_cast<std::size_t>(i)] != 0)
                nz.push_back(probe[static_cast<std::size_t>(i)]);
            std::fill(probe.begin() + lo, probe.begin() + hi, 0);
            for (std::size_t t = 0; t < nz.size(); ++t)
              probe[static_cast<std::size_t>(lo) + t] = nz[t];
          }
          GrInfo t4{};
          packed[k] += static_cast<double>(final_bits(probe.data(), t4, s)) /
                       static_cast<double>(full);
        }
      }
      static long long tick = 0;
      if ((++tick % 2000) == 0) {
        std::fprintf(stderr, "[bandcost] n=%lld", n);
        for (int k = 0; k < 5; ++k)
          std::fprintf(stderr, "  sfb%d+: cost %.1f%% clamp %.0f%% packed %.0f%%",
                       probes[k], 100.0 * frac[k] / static_cast<double>(n),
                       100.0 * cheap[k] / static_cast<double>(n),
                       100.0 * packed[k] / static_cast<double>(n));
        std::fprintf(stderr, "\n");
      }
    }
    if (hf_saved) {
      // best.bits / cum_dist already updated by hf_rdo via reference args.
      best.cum_dist = 0.0;
      for (double d : best.dist_band)
        best.cum_dist += d;
    }

    // --- High-frequency region post-pass (sfb 21, ~16–20 kHz at 44.1k) ---
    // The beam only allocates the 21 transmitted scalefactor bands (up to
    // band_table[NUM_SFB_LONG] ≈ 16 kHz). The region above is legal MP3
    // content coded by the big_values/count1 machinery with an *implicit*
    // scalefactor of 0 — only the global gain applies and nothing extra is
    // sent in the side info. Quantise it at the chosen global gain and keep
    // it whenever it still fits the bit budget, so genuine >16 kHz content
    // (e.g. from hi-res 24/32-bit masters) survives instead of being hard-
    // zeroed. When the band is silent the quantizer yields all zeros and the
    // splice costs nothing; when the budget is tight the candidate is
    // reverted, preserving the previous behaviour at low bitrates.
    double hf_raw_dist = 0.0;
    if (hf_start < hf_end) {
      const std::int64_t scalei_global = s.steptabi[best.gain + 127];
      const double steptab_global = s.steptab[best.gain + 127];
      auto q = quantize_band(s.xrabs.data() + hf_start, hf_end - hf_start,
                             scalei_global, steptab_global,
                             /*sf=*/0, best.scale, best.preflag,
                             /*sfb=*/NUM_SFB_LONG, s.int2idx, eff_nsa,
                             zb_factor_hf,
                             zb_fill_exp,
                             prev_base ? prev_base + hf_start : nullptr, hyst_scale);
      bool used_ext_hf = false;
      if (s.cfg_hf_extension) {
        bool has_ext = false;
        for (int j = 0; j < hf_end - hf_start; ++j) {
          if (xrabs_ext[hf_start + j] > 0) {
            has_ext = true;
            break;
          }
        }
        if (has_ext) {
          auto q_ext = quantize_band(
              xrabs_ext.data() + hf_start, hf_end - hf_start, scalei_global,
              steptab_global, /*sf=*/0, best.scale, best.preflag,
              /*sfb=*/NUM_SFB_LONG, s.int2idx, eff_nsa, zb_factor_hf, zb_fill_exp,
              prev_base ? prev_base + hf_start : nullptr, hyst_scale);
          if (q_ext.ok && q_ext.max_val > 0) {
            if (q.max_val == 0 || s.hf_cutoff_sfb <= NUM_SFB_LONG) {
              q = q_ext;
              used_ext_hf = true;
            }
          }
        }
      }
      if (q.ok && q.max_val > 0 && !used_ext_hf) {
        // Over-shoot guard for the splice. This region never passes through
        // `hybrid_band_cost`, so the HF over-shoot charge (HF_OVER_WEIGHT)
        // does not see it — and it is the region most prone to running hot,
        // because it is coded at an implicit sf of 0: every bin the shaping
        // quantizer promotes out of the dead zone lands at the |ix|=1
        // reconstruction level whether or not the source was that loud.
        // Emitting the region is the point (that is what the reservation
        // above buys), emitting *more than was there* is not: without this,
        // mixed_music@192 went from −0.29 to +3.00 dB of >8 kHz balance.
        // Prune the promoted bins, quietest source first, until the region's
        // reconstructed energy is back inside the same +1.5 dB dead-band the
        // per-band charge uses.
        {
          double e_src = 0.0, e_rec = 0.0;
          for (int j = 0; j < hf_end - hf_start; ++j) {
            const double v = s.xr[hf_start + j];
            e_src += v * v;
            const double r =
                tables::pow43_tab[q.ix[static_cast<std::size_t>(j)]] *
                q.dequant_factor;
            e_rec += r * r;
          }
          const double allowance = e_src * HF_OVER_DEADBAND;
          if (e_rec > allowance) {
            // Candidates are the |ix|=1 bins — the only ones whose removal
            // costs a single reconstruction level rather than a real value.
            std::vector<std::pair<std::int32_t, int>> cand;
            for (int j = 0; j < hf_end - hf_start; ++j)
              if (q.ix[static_cast<std::size_t>(j)] == 1)
                cand.emplace_back(s.xrabs[hf_start + j], j);
            std::sort(cand.begin(), cand.end());
            const double unit = q.dequant_factor * q.dequant_factor;
            for (const auto &[mag, j] : cand) {
              if (e_rec <= allowance)
                break;
              (void)mag;
              q.ix[static_cast<std::size_t>(j)] = 0;
              e_rec -= unit;
            }
            q.max_val = 0;
            for (int j = 0; j < hf_end - hf_start; ++j)
              q.max_val = std::max(q.max_val, q.ix[static_cast<std::size_t>(j)]);
          }
        }
      }
      if (q.ok && q.max_val > 0) {
        for (int j = 0; j < hf_end - hf_start; ++j)
          best.ix[static_cast<std::size_t>(hf_start + j)] =
              q.ix[static_cast<std::size_t>(j)];
        GrInfo hf_gi{};
        int hf_bits = final_bits(best.ix.data(), hf_gi, s);
        if (hf_bits > max_bits - best.part2_length) {
          for (int j = hf_end - hf_start - 1; j >= 0; --j) {
            if (q.ix[static_cast<std::size_t>(j)] > 0) {
              q.ix[static_cast<std::size_t>(j)] = 0;
              best.ix[static_cast<std::size_t>(hf_start + j)] = 0;
              hf_bits = final_bits(best.ix.data(), hf_gi, s);
              if (hf_bits <= max_bits - best.part2_length)
                break;
            }
          }
          q.max_val = 0;
          for (int j = 0; j < hf_end - hf_start; ++j)
            q.max_val = std::max(q.max_val, q.ix[static_cast<std::size_t>(j)]);
        }
        if (std::getenv("RMPS_HF21_DEBUG"))
          std::fprintf(stderr, "[hf21] need %d have %d slack %d %s (used_ext=%d)\n", hf_bits,
                       max_bits - best.part2_length,
                       max_bits - best.part2_length - hf_bits,
                       (hf_bits <= max_bits - best.part2_length) ? "KEEP"
                                                                 : "DROP",
                       (int)used_ext_hf);
        if (q.max_val > 0 && hf_bits <= max_bits - best.part2_length) {
          best.tmp_gi = hf_gi;
          best.bits = hf_bits;
          double e_recon = 0.0;
          const std::int32_t *ref_xr =
              used_ext_hf ? (s.mdct_freq_ext[ch][gr].data() + hf_start)
                          : (s.xr + hf_start);
          hf_raw_dist = band_distortion(ref_xr, q.ix, q.dequant_factor,
                                        &e_recon);
          if (used_ext_hf) {
            for (int j = 0; j < hf_end - hf_start; ++j) {
              if (q.ix[static_cast<std::size_t>(j)] > 0) {
                s.mdct_freq[ch][gr][hf_start + j] =
                    s.mdct_freq_ext[ch][gr][hf_start + j];
              }
            }
          }
        } else {
          for (int j = 0; j < hf_end - hf_start; ++j)
            best.ix[static_cast<std::size_t>(hf_start + j)] = 0;
        }
      }
    }

    // --- Temporal rounding polish (pre-echo shaping) ---
    // MP3 has no TNS: quantization noise of a long-block granule spreads
    // over the whole 36-sample (subband-time) MDCT window, and the part
    // that lands *before* an attack decodes as pre-echo. One degree of
    // freedom is still free after the beam has fixed gain/scalefactors:
    // the rounding direction of each coefficient. Nearest rounding
    // minimises spectral MSE; nudging selected coefficients by ±1
    // quantization level re-phases the error waveform so it lands mostly
    // *after* the attack, where forward masking hides it — a TNS-like
    // temporal noise control that stays fully bitstream-compatible.
    //
    // Scope: granules whose MDCT window actually contains the attack, on a
    // long-family window (bass-heavy hits kept long by the HF gate, or
    // --no-short-blocks) — exactly where pre-echo is otherwise only
    // mitigated by cost weighting.
    //
    // The gate's attack_sub_index is a raw-PCM position, but the MDCT sees
    // the signal only after the 512-tap polyphase analysis (≈256-sample
    // group delay = 8 subband-time samples). Correct for it — otherwise
    // the protected region ends ~5.8 ms *before* the real attack and the
    // shaping dumps error right into the most exposed pre-attack zone.
    // With the correction, a late attack (sub-window 2) actually lands in
    // the NEXT granule's window, so the spill-over case is handled by
    // looking one granule (possibly one frame) back.
    int tns_cut = -1;
    {
      constexpr int kPolyDelaySub = 8; // 256 / 32 subband-time samples
      const auto win_pos = [&](int sub) {
        return 6 * sub + kPolyDelaySub; // window offset past sample 18
      };
      if ((s.transient_flags[gr] || s.attack_from_silence_flags[gr]) &&
          s.attack_sub_index[gr] >= 0) {
        const int o = win_pos(s.attack_sub_index[gr]);
        if (o < 18)
          tns_cut = 18 + o + 1; // +1: safety margin
      }
      if (tns_cut < 0) {
        bool prev_attack = false;
        int prev_sub = -1;
        if (gr > 0) {
          prev_attack =
              s.transient_flags[gr - 1] || s.attack_from_silence_flags[gr - 1];
          prev_sub = s.attack_sub_index[gr - 1];
        } else {
          prev_attack = s.prev_frame_attack;
          prev_sub = s.prev_frame_attack_sub;
        }
        if (prev_attack && prev_sub >= 0) {
          const int o = win_pos(prev_sub);
          if (o >= 18)
            tns_cut = std::min(35, o + 1);
        }
      }
      if (std::getenv("RMPS_TNS_DEBUG"))
        std::fprintf(stderr,
                     "[tns-gate] gr=%d ch=%d tr=%d afs=%d sub=%d prevfr(att=%d "
                     "sub=%d) bt=%d -> cut=%d\n",
                     gr, ch, (int)s.transient_flags[gr],
                     (int)s.attack_from_silence_flags[gr],
                     s.attack_sub_index[gr], (int)s.prev_frame_attack,
                     s.prev_frame_attack_sub, s.block_type[ch][gr], tns_cut);
    }
    if (tns_cut >= 0 && s.block_type[ch][gr] != BLOCK_SHORT) {
      constexpr double kPi = 3.14159265358979323846;
      const int bt = s.block_type[ch][gr];
      // 36x18 windowed synthesis basis for this granule's window shape:
      // an error ΔX_m contributes ΔX_m · B[k][m] to subband-time sample k.
      std::array<std::array<double, 18>, 36> B;
      for (int k = 0; k < 36; ++k) {
        double w;
        if (bt == BLOCK_START) {
          if (k < 18)
            w = std::sin(kPi / 36.0 * (k + 0.5));
          else if (k < 24)
            w = 1.0;
          else if (k < 30)
            w = std::sin(kPi / 12.0 * (k - 18 + 0.5));
          else
            w = 0.0;
        } else if (bt == BLOCK_STOP) {
          if (k < 6)
            w = 0.0;
          else if (k < 12)
            w = std::sin(kPi / 12.0 * (k - 6 + 0.5));
          else if (k < 18)
            w = 1.0;
          else
            w = std::sin(kPi / 36.0 * (k + 0.5));
        } else {
          w = std::sin(kPi / 36.0 * (k + 0.5));
        }
        for (int m = 0; m < 18; ++m)
          B[k][m] =
              w * std::cos(kPi / 72.0 * (2.0 * k + 19.0) * (2.0 * m + 1.0));
      }
      // Pre-attack region: window samples [0, cut) — everything the
      // window covers up to the (delay-corrected) attack. The 512-tap
      // polyphase *synthesis* filter pre-rings each subband-time sample
      // by up to 8 samples (≈5.8 ms), so error parked immediately after
      // the attack still leaks back in front of it: only k ≳ cut+5 is a
      // safe dump zone. Ramp the weight down across that guard span;
      // beyond it forward masking hides the noise.
      const int cut = tns_cut;
      // Shape of the temporal penalty, swept via env because these three
      // numbers decide how hard the polish works and none had been measured:
      //   RMPS_TNS_W_PRE   weight on error landing before the attack (1.2)
      //   RMPS_TNS_W_POST  weight after it (0.08)
      //   RMPS_TNS_ECAP    how much spectral error may grow to buy that (1.35)
      //   RMPS_TNS_PASSES  sweeps over the 18 coefficients (2)
      static const double w_pre = [] {
        if (const char *e = std::getenv("RMPS_TNS_W_PRE")) {
          const double v = std::atof(e);
          if (v > 0.0 && v <= 100.0) return v;
        }
        return 1.2;
      }();
      static const double w_post = [] {
        if (const char *e = std::getenv("RMPS_TNS_W_POST")) {
          const double v = std::atof(e);
          if (v >= 0.0 && v <= 100.0) return v;
        }
        return 0.08;
      }();
      std::array<double, 36> W;
      for (int k = 0; k < 36; ++k) {
        if (k < cut)
          W[k] = w_pre;
        else if (k < cut + 5)
          W[k] = w_pre - (w_pre - w_post) / 5.0 * (k - cut + 1);
        else
          W[k] = w_post;
      }

      // Per-coefficient dequant factor from the chosen gain/sf/scale.
      std::array<double, GRANULE_SAMPLES> dq{};
      {
        const std::int64_t sg = s.steptabi[best.gain + 127];
        for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
          const int eff_sf =
              best.sf[static_cast<std::size_t>(sfb)] +
              (best.preflag ? PRETAB[static_cast<std::size_t>(sfb)] : 0);
          const double sf_mult = tables::pow2_half_tab[static_cast<std::size_t>(
              (1 + best.scale) * eff_sf)];
          const std::int64_t scalei = static_cast<std::int64_t>(
              static_cast<double>(sg) * sf_mult + 0.5);
          const double d =
              (scalei > 0) ? 4294967296.0 / static_cast<double>(scalei) : 0.0;
          for (int i = band_table[sfb]; i < band_table[sfb + 1]; ++i)
            dq[static_cast<std::size_t>(i)] = d;
        }
        const double d0 =
            (sg > 0) ? 4294967296.0 / static_cast<double>(sg) : 0.0;
        for (int i = band_table[NUM_SFB_LONG]; i < GRANULE_SAMPLES; ++i)
          dq[static_cast<std::size_t>(i)] = d0;
      }

      const auto saved_ix = best.ix;
      bool changed = false;
      double dbg_J0 = 0.0, dbg_J1 = 0.0;
      int dbg_flips = 0;
      // Flip journal, in acceptance order (best dJ first within a pass) —
      // lets the bit-budget repair below undo the least valuable flips
      // instead of discarding the whole polish over a few bits.
      std::array<std::pair<int, int>, 128> flip_log{}; // (index, old_v)
      int n_flips = 0;
      // Coefficients inside the <200 Hz bass-protection bands are never
      // re-rounded: their waveform fidelity outranks temporal shaping.
      // Everything above (including the rest of subband 0 — a kick's
      // 200–700 Hz body is a major pre-echo contributor) is fair game.
      int i_min = GRANULE_SAMPLES;
      for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const double lo_hz = band_table[sfb] * s.wave.samplerate / 1152.0;
        if (lo_hz >= 200.0) {
          i_min = band_table[sfb];
          break;
        }
      }
      for (int sb = 0; sb < 32; ++sb) {
        const int base = sb * 18;
        std::array<double, 18> err{};
        std::array<double, 36> t{};
        double e_tot = 0.0;
        bool any_coded = false;
        for (int m = 0; m < 18; ++m) {
          const int i = base + m;
          const int v = best.ix[static_cast<std::size_t>(i)];
          if (v > 0)
            any_coded = true;
          const double x = static_cast<double>(s.xr[i]);
          double xh = tables::pow43_tab[v] * dq[static_cast<std::size_t>(i)];
          if (x < 0.0)
            xh = -xh;
          err[static_cast<std::size_t>(m)] = xh - x;
          e_tot += err[static_cast<std::size_t>(m)] *
                   err[static_cast<std::size_t>(m)];
        }
        if (!any_coded || e_tot <= 0.0)
          continue;
        for (int k = 0; k < 36; ++k) {
          double acc = 0.0;
          for (int m = 0; m < 18; ++m)
            acc += err[static_cast<std::size_t>(m)] * B[k][m];
          t[static_cast<std::size_t>(k)] = acc;
        }
        for (int k = 0; k < 36; ++k)
          dbg_J0 += W[k] * t[static_cast<std::size_t>(k)] *
                    t[static_cast<std::size_t>(k)];
        // Total (unweighted) error may grow while shaping — keep it strictly bounded
        // to max 1.35x of total energy so overall quantization noise does not bloat.
        static const double ecap_f = [] {
          if (const char *e = std::getenv("RMPS_TNS_ECAP")) {
            const double v = std::atof(e);
            if (v >= 1.0 && v <= 100.0) return v;
          }
          return 1.35;
        }();
        const double e_cap = e_tot * ecap_f;
        static const int n_passes = [] {
          if (const char *e = std::getenv("RMPS_TNS_PASSES")) {
            const int v = std::atoi(e);
            if (v >= 1 && v <= 16) return v;
          }
          return 2;
        }();
        for (int pass = 0; pass < n_passes; ++pass) {
          bool improved = false;
          for (int m = 0; m < 18; ++m) {
            const int i = base + m;
            if (i < i_min)
              continue; // <200 Hz protection bands
            const int v = best.ix[static_cast<std::size_t>(i)];
            // v==0 with a nonzero source coefficient may be promoted
            // to 1 (that moves the reconstruction TOWARD the source,
            // it doesn't fabricate content) — the dominant pre-echo
            // term is precisely the error of the zeroed lines, and
            // re-rounding only coded lines leaves it untouchable.
            if (v < 1 && s.xr[i] == 0)
              continue;
            const double sign = (s.xr[i] < 0) ? -1.0 : 1.0;
            for (int vd : {-1, +1}) {
              const int v2 = v + vd;
              if (v2 < 0 || v2 > 8191)
                continue;
              const double delta =
                  sign * (tables::pow43_tab[v2] - tables::pow43_tab[v]) *
                  dq[static_cast<std::size_t>(i)];
              const double e_new =
                  e_tot +
                  delta * (2.0 * err[static_cast<std::size_t>(m)] + delta);
              if (e_new > e_cap)
                continue;
              double dJ = 0.0;
              for (int k = 0; k < 36; ++k) {
                const double bk = B[k][m];
                dJ += W[k] * delta * bk *
                      (2.0 * t[static_cast<std::size_t>(k)] + delta * bk);
              }
              if (dJ >= 0.0)
                continue;
              for (int k = 0; k < 36; ++k)
                t[static_cast<std::size_t>(k)] += delta * B[k][m];
              err[static_cast<std::size_t>(m)] += delta;
              e_tot = e_new;
              if (n_flips < static_cast<int>(flip_log.size()))
                flip_log[static_cast<std::size_t>(n_flips++)] = {i, v};
              best.ix[static_cast<std::size_t>(i)] = v2;
              changed = true;
              improved = true;
              ++dbg_flips;
              break;
            }
          }
          if (!improved)
            break;
        }
        for (int k = 0; k < 36; ++k)
          dbg_J1 += W[k] * t[static_cast<std::size_t>(k)] *
                    t[static_cast<std::size_t>(k)];
      }
      if (changed) {
        GrInfo pg{};
        int pb = final_bits(best.ix.data(), pg, s);
        // Bit-budget repair: undo flips from the tail (weakest gains)
        // until the granule fits again, rather than losing the whole
        // polish over a handful of bits.
        while (pb > max_bits - best.part2_length && n_flips > 0) {
          const auto [idx, old_v] =
              flip_log[static_cast<std::size_t>(--n_flips)];
          best.ix[static_cast<std::size_t>(idx)] = old_v;
          pb = final_bits(best.ix.data(), pg, s);
        }
        const bool fits = pb <= max_bits - best.part2_length;
        if (std::getenv("RMPS_TNS_DEBUG"))
          std::fprintf(
              stderr,
              "[tns] gr=%d ch=%d cut=%d flips=%d J %.3e->%.3e (%.1f%%) "
              "bits %d->%d budget %d %s\n",
              gr, ch, cut, dbg_flips, dbg_J0, dbg_J1,
              (dbg_J0 > 0) ? 100.0 * dbg_J1 / dbg_J0 : 0.0, best.bits, pb,
              max_bits - best.part2_length, fits ? "KEEP" : "REVERT");
        if (fits) {
          best.tmp_gi = pg;
          best.bits = pb;
        } else {
          best.ix = saved_ix; // shaped rounding didn't fit: revert
        }
      } else if (std::getenv("RMPS_TNS_DEBUG")) {
        std::fprintf(stderr, "[tns] gr=%d ch=%d cut=%d no-moves\n", gr, ch,
                     cut);
      }
    }

    // --- Black-spot guard ---
    // A granule whose every coefficient quantises to zero decodes as a silent
    // frame — an audible spectral hole / dropout ("黒点"). Budget starvation
    // and the coarse fallback extreme (gain→127) can both reach this state. If
    // the source granule actually carries energy, force the single most
    // audible coded bin to ix=1: a few count1 bits the reservoir can always
    // absorb, turning a dropout into a faint-but-present granule. This cannot
    // fire on normal granules (they already have non-zero coefficients), so it
    // is inert outside true starvation.
    if (s.xrmax > 0) {
      bool all_zero = true;
      for (int i = 0; i < GRANULE_SAMPLES; ++i)
        if (best.ix[i] != 0) {
          all_zero = false;
          break;
        }
      if (all_zero) {
        // Choosing the bin by perceptual weight alone is what made this guard
        // audible. `ix = 1` does not reconstruct to the coefficient's own
        // magnitude — it reconstructs to this band's `dequant_factor`, i.e. the
        // bottom of the codebook at whatever step the granule ended on. The
        // granule is all-zero precisely because that step is coarser than every
        // coefficient in it, so the injected value is *always* above the source
        // and can be far above it: measured on `avicii` at 64 kbps, isolated
        // bursts of +11 to +17 dB over the source's own local peak, in quiet
        // passages, at 20x the rate the source itself has such peaks. They read
        // as a click on top of whatever is playing.
        //
        // The weight made it worse rather than caused it: weighting pulls the
        // choice toward a high band, and the highest-weighted bin is rarely the
        // largest one, so the guard reliably picked a bin with *more* headroom
        // to overshoot than the granule's loudest.
        //
        // So score by overshoot first and audibility second: among bins whose
        // reconstruction lands within `MAX_OVERSHOOT_DB` of their own source
        // magnitude, take the most perceptually important. If no bin qualifies,
        // leave the granule silent — a granule whose loudest coefficient is
        // more than this far under the smallest codeable value is quiet enough
        // that a dropout is the lesser artefact, and injecting there is trading
        // an inaudible hole for an audible click.
        static const double max_over_db = [] {
          if (const char *e = std::getenv("RMPS_RESCUE_OVER_DB")) {
            const double v = std::atof(e);
            if (v >= 0.0 && v <= 60.0) return v;
          }
          return 9.0;
        }();
        const double max_over = std::pow(10.0, max_over_db / 20.0);
        const std::int64_t sg_guard = s.steptabi[best.gain + 127];
        int floor_bin = -1;
        double floor_score = 0.0;
        for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
          const int start = band_table[sfb];
          const double w = weights[static_cast<std::size_t>(sfb)];
          // Reconstruction magnitude of |ix| = 1 in this band, i.e. the level
          // the injection would actually produce. Same chain quantize_band
          // uses: the granule step scaled by this band's scalefactor.
          const int eff_sf =
              best.sf[static_cast<std::size_t>(sfb)] +
              (best.preflag ? PRETAB[static_cast<std::size_t>(sfb)] : 0);
          const std::size_t sf_idx =
              static_cast<std::size_t>((1 + best.scale) * eff_sf);
          if (sf_idx >= tables::pow2_half_tab.size()) continue;
          const std::int64_t scalei = static_cast<std::int64_t>(
              static_cast<double>(sg_guard) * tables::pow2_half_tab[sf_idx] +
              0.5);
          if (scalei <= 0) continue;
          const double recon = 4294967296.0 / static_cast<double>(scalei);
          for (int j = 0; j < band_lens[sfb]; ++j) {
            const double mag = static_cast<double>(xrabs_bands[sfb][j]);
            if (mag <= 0.0) continue;
            if (recon > mag * max_over) continue;   // would click
            const double sc = mag * w;
            if (sc > floor_score) {
              floor_score = sc;
              floor_bin = start + j;
            }
          }
        }
        if (floor_bin >= 0) {
          best.ix[static_cast<std::size_t>(floor_bin)] = 1;
          GrInfo fg{};
          const int fb = final_bits(best.ix.data(), fg, s);
          // Keep the bin unconditionally: a single count1 coefficient is
          // only a few bits, never approaches the 4095-bit hard granule
          // cap, and any overshoot of the (soft) per-granule share flows
          // through the cross-frame reservoir. Reverting here would
          // re-introduce the very dropout this guard exists to remove, so
          // black-spot avoidance wins over the advisory bit target.
          best.tmp_gi = fg;
          best.bits = fb;
        }
      }
    }

    // Rate-model diagnostic (RMPS_RATE_DEBUG=1): compares the old lower bound,
    // the Huffman estimate and the real coded size for the granule the search
    // actually chose. `est ≈ real` is the property the RD term needs; `lb` is
    // what the cost used to be priced with.
    if (std::getenv("RMPS_RATE_DEBUG")) {
      int lb = 0, est = 0, mrg = 0;
      for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        int *b = best.ix.data() + band_table[sfb];
        const int n = band_lens[sfb];
        for (int i = 0; i < n; i += 2) {
          const int x = b[i], y = b[i + 1];
          if (x || y)
            ++lb;
          if (x)
            ++lb;
          if (y)
            ++lb;
          if (x > 14)
            ++lb;
          if (y > 14)
            ++lb;
        }
        const BandRate r = band_rate(b, n);
        est += r.total;
        mrg += r.marginal;
      }
      std::fprintf(stderr, "[rate] lb=%d est=%d mrg=%d real=%d\n", lb, est, mrg,
                   best.bits);
    }

    // --- Compute raw distortion for stats ---
    double raw_dist = 0.0;
    double weighted_dist = 0.0;
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
      const double d = best.raw_band[static_cast<std::size_t>(sfb)];
      raw_dist += d;
      weighted_dist += d * adj_weights[sfb];
    }
    raw_dist += hf_raw_dist;
    weighted_dist += hf_raw_dist * weights[NUM_SFB_LONG - 1];
    out.raw_dist = raw_dist;
    out.weighted_dist = weighted_dist;

    // Fullness / Bleedless for this granule, in dB, both <= 0.
    //
    // Same shape as tools/fbscore.py: the two ways of being wrong are scored
    // separately and never allowed to compensate. `def_band` is source energy
    // the coding failed to keep, `exc_band` is energy it invented; measured
    // against the same A-weighted signal power the SDR uses, so the numbers
    // sit on a comparable scale.
    {
      double wdef = 0.0, wexc = 0.0;
      for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        if (sfb < static_cast<int>(best.def_band.size())) {
          wdef += best.def_band[static_cast<std::size_t>(sfb)] * adj_weights[sfb];
          wexc += best.exc_band[static_cast<std::size_t>(sfb)] * adj_weights[sfb];
        }
      }
      // Expressed as signal-to-failure ratios, the same shape and units as
      // SDR — larger is better, tens of dB of range. The first attempt used
      // `-10*log10(1 + w/sig)`, which is right in spirit but collapses:
      // that form is ~-4.34*(w/sig) for small w, so a granule at 30 dB SDR
      // scored -0.004 dB and the whole file spanned -0.25 to 0.00. Nothing
      // can be a PID setpoint with that little range.
      //
      // Keeping SDR's units is deliberate: the difference from SDR is not the
      // scale but *what is measured*. SDR sums the two failures; these two
      // keep them apart, and the controller is handed the worse one, so a
      // frame cannot earn its bitrate by being full while it bleeds.
      const double sig = std::max(out.weighted_signal_pow, 1.0);
      out.fullness_db  = 10.0 * std::log10(sig / std::max(wdef, 1.0));
      out.bleedless_db = 10.0 * std::log10(sig / std::max(wexc, 1.0));
      out.weighted_def = wdef;
      out.weighted_exc = wexc;
    }
    out.cum_dist = best.cum_dist + hf_raw_dist * weights[NUM_SFB_LONG - 1];

    if (std::getenv("RMPS_EREF_DEBUG")) {
      double xmin = 1e300, xmax = 0.0, xsum = 0.0;
      int nb = 0;
      for (int b = 0; b < NUM_SFB_LONG; ++b) {
        const double x = best.raw_band[static_cast<std::size_t>(b)] /
                         std::max(e_ref, 1.0);
        if (x > 0.0) {
          xmin = std::min(xmin, x);
          xmax = std::max(xmax, x);
          xsum += x;
          ++nb;
        }
      }
      if (nb)
        std::fprintf(stderr, "[xabs] min=%.3e mean=%.3e max=%.3e\n", xmin,
                     xsum / nb, xmax);
    }
    if (std::getenv("RMPS_SF_DEBUG")) {
      int mx = 0, nz = 0;
      for (int b = 0; b < NUM_SFB_LONG; ++b) {
        mx = std::max(mx, static_cast<int>(best.sf[static_cast<std::size_t>(b)]));
        if (best.sf[static_cast<std::size_t>(b)] > 0)
          ++nz;
      }
      int hi_mx = 0;
      for (int b = 11; b < NUM_SFB_LONG; ++b)
        hi_mx = std::max(
            hi_mx, static_cast<int>(best.sf[static_cast<std::size_t>(b)]));
      std::fprintf(stderr,
                   "[sf] compress=%d sf_max_cfg=%d maxsf=%d nonzero=%d "
                   "preflag=%d hi_maxsf=%d\n",
                   best.compress, sf_max, mx, nz, best.preflag, hi_mx);
    }

    // --- Apply best to cod_info / scalefactor / ix output ---
    cod.scalefac_compress = best.compress;
    cod.scalefac_scale = best.scale;
    cod.preflag = best.preflag;
    cod.quantizerStepSize = best.gain;
    cod.part2_length = best.part2_length;
    if (gr == 1 && gr0_sf != nullptr && mpeg1) {
      for (int b = 0; b < 4; ++b)
        s.side_info.scfsi[ch][b] = best.scfsi_pat[b];
    }
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb)
      s.scalefactor.l[gr][ch][sfb] = best.sf[static_cast<std::size_t>(sfb)];
    for (int i = 0; i < GRANULE_SAMPLES; ++i)
      ix[i] = best.ix[i];

    if (s.cfg_hf_extension) {
      int ext_replaced_bands = 0;
      for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const double lo_hz = band_table[sfb] * s.wave.samplerate / 1152.0;
        if (lo_hz > 20000.0)
          break;
        if (best.is_ext[static_cast<std::size_t>(sfb)] ||
            (sfb >= s.hf_cutoff_sfb && has_ext_hf[sfb])) {
          const int start = band_table[sfb];
          bool has_nonzero = false;
          for (int j = 0; j < band_lens[sfb]; ++j) {
            if (best.ix[start + j] != 0) {
              has_nonzero = true;
              break;
            }
          }
          if (has_nonzero) {
            ++ext_replaced_bands;
            for (int j = 0; j < band_lens[sfb]; ++j) {
              s.mdct_freq[ch][gr][start + j] =
                  s.mdct_freq_ext[ch][gr][start + j];
            }
          }
        }
      }
      static const bool hf_dbg = (std::getenv("RMPS_HF_DEBUG") != nullptr);
      if (hf_dbg && ext_replaced_bands > 0) {
        std::fprintf(stderr,
                     "[hf_ext] f=%lld gr=%d ch=%d replaced %d SFBs "
                     "(cutoff_sfb=%d)\n",
                     static_cast<long long>(s.dbg_frame_no), gr, ch,
                     ext_replaced_bands, s.hf_cutoff_sfb);
        if (s.dbg_frame_no == 50 && gr == 0 && ch == 0) {
          for (int sfb = s.hf_cutoff_sfb; sfb <= NUM_SFB_LONG; ++sfb) {
            int start = (sfb < NUM_SFB_LONG) ? band_table[sfb] : band_table[NUM_SFB_LONG];
            int end = (sfb < NUM_SFB_LONG) ? band_table[sfb + 1] : 576;
            int nz = 0, mx = 0;
            for (int j = start; j < end; ++j) {
              if (best.ix[j] != 0) { ++nz; mx = std::max(mx, std::abs(best.ix[j])); }
            }
            int sf_val = (sfb < NUM_SFB_LONG) ? s.scalefactor.l[gr][ch][sfb] : 0;
            std::fprintf(stderr, "[debug_sfb] sfb=%d: nz=%d/%d max_ix=%d sf=%d gain=%d\n",
                         sfb, nz, end - start, mx, sf_val, best.gain);
          }
        }
      }
    }

    cod.big_values = best.tmp_gi.big_values;
    cod.count1 = best.tmp_gi.count1;
    cod.count1table_select = best.tmp_gi.count1table_select;
    cod.address1 = best.tmp_gi.address1;
    cod.address2 = best.tmp_gi.address2;
    cod.address3 = best.tmp_gi.address3;
    cod.region0_count = best.tmp_gi.region0_count;
    cod.region1_count = best.tmp_gi.region1_count;
    cod.table_select = best.tmp_gi.table_select;

    // --- RD-optimal Huffman region split (finalisation-only) ---
    // The beam ranks candidates with the fast fixed-table subdivide; once the
    // winner is chosen, re-split its big-values region optimally. This is a
    // pure bit reclaim: the recount can only shrink part2_3_length, and every
    // bit saved flows into the cross-frame reservoir.
    {
      // `best_runlen_rd` also moves the big_values / count1 boundary, which
      // the beam fixes greedily via `calc_runlen` and never revisits. Same
      // argument as the region split: it can only shrink part2_3_length,
      // because the greedy boundary is one of the candidates it compares.
      cod.part2_3_length = best.part2_length + best_runlen_rd(ix, cod, s);
      // Debug snapshot for the writer's stale-side-info check
      // (RMPS_BS_DEBUG=1); off by default, it costs a copy per granule.
      static const bool bs_debug = std::getenv("RMPS_BS_DEBUG") != nullptr;
      if (bs_debug) {
        auto &snap = s.dbg_fin_ix[static_cast<std::size_t>(gr)]
                                 [static_cast<std::size_t>(ch)];
        for (int i = 0; i < GRANULE_SAMPLES; ++i)
          snap[static_cast<std::size_t>(i)] = std::abs(ix[i]);
        s.dbg_fin_bigv[gr][ch] = cod.big_values;
        s.dbg_fin_count1[gr][ch] = cod.count1;
        s.dbg_fin_p23[gr][ch] = cod.part2_3_length;
        s.dbg_fin_valid[gr][ch] = true;
      }
    }

    out.ok = true;
    out.part2_3_length = cod.part2_3_length;
    return out;
  }

} // namespace rmps::quant
