#include "quant/short_block.hpp"
#include "quant/quantize.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include "core/tables.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace rmps::quant {

namespace {

// A contiguous run of coefficients in short transmission order belonging to one
// (scalefactor band, window). Transmission order is grouped (sfb, window, line),
// so each (sfb, w) maps to one contiguous [begin, begin+len) range.
struct Seg {
  int sfb;
  int w;
  int begin;
  int len;
};

int build_segments(const EncoderState& s, std::array<Seg, tables::NUM_SCFB_SHORT * NUM_WINDOWS>& segs) {
  const auto& sb_s = tables::scale_fact_band_index_short[s.mpeg.samplerate_index];
  int n = 0;
  int pos = 0;
  for (int sfb = 0; sfb < tables::NUM_SCFB_SHORT - 1; ++sfb) {
    const int width = sb_s[sfb + 1] - sb_s[sfb];
    for (int w = 0; w < NUM_WINDOWS; ++w) {
      segs[static_cast<std::size_t>(n)] = Seg{sfb, w, pos, width};
      pos += width;
      ++n;
    }
  }
  return n; // == 39, total coefficients == GRANULE_SAMPLES
}

// Effective per-band gain index folding scalefactor and subblock_gain into the
// global gain: every +1 of sf attenuates by 2^(0.5(1+scale)) = 2*(1+scale)
// table steps; every +1 of subblock_gain attenuates by 2^2 = 8 table steps.
inline int eff_gain(int gain, int scale, int sf, int sbg) {
  int g = gain - 2 * (1 + scale) * sf - 8 * sbg;
  return std::clamp(g, -127, 127);
}

// Quantize all 576 coefficients in short transmission order with per-(w,sfb)
// scalefactors and per-window subblock_gain. Returns max |ix|.
int quantize_all(int* ix, int gain, int scale,
                 const std::array<std::array<int, NUM_SFB_SHORT>, NUM_WINDOWS>& sf,
                 const std::array<int, NUM_WINDOWS>& sbg,
                 const std::array<Seg, tables::NUM_SCFB_SHORT * NUM_WINDOWS>& segs, int nseg,
                 EncoderState& s) {
  int maxv = 0;
  for (int si = 0; si < nseg; ++si) {
    const Seg& sg = segs[static_cast<std::size_t>(si)];
    const int sfv = (sg.sfb < NUM_SFB_SHORT) ? sf[static_cast<std::size_t>(sg.w)][static_cast<std::size_t>(sg.sfb)] : 0;
    const int g = eff_gain(gain, scale, sfv, sbg[static_cast<std::size_t>(sg.w)]);
    const std::int32_t scalei = s.steptabi[static_cast<std::size_t>(g + 127)];
    for (int i = sg.begin; i < sg.begin + sg.len; ++i) {
      const std::int32_t x = s.xrabs[static_cast<std::size_t>(i)];
      const std::int32_t ln = fx::mulr(x, scalei);
      int v;
      if (ln >= 0 && ln < 262144)
        v = s.int2idx[static_cast<std::size_t>(ln)];
      else
        v = 8191;
      if (v > 8191) v = 8191;
      ix[i] = v;
      if (v > maxv) maxv = v;
    }
  }
  return maxv;
}

// Short-block Huffman bit count + side-info region fields. The big-values
// region split is fixed at coefficient 36 (region0 = [0,36), region1 =
// [36, 2*big_values)); table_select[2] is unused. Reuses the layout-agnostic
// calc_runlen / count1_bitcount / table-select helpers.
int short_huffman_bits(int* ix, GrInfo& gi) {
  calc_runlen(ix, gi);
  const int c1 = count1_bitcount(ix, gi);

  const int bigv = gi.big_values << 1;
  const int a1 = std::min(36, bigv);
  gi.address1 = a1;
  gi.address2 = bigv;
  gi.address3 = bigv;
  gi.region0_count = 0; // not transmitted for window_switching; kept for clarity
  gi.region1_count = 0;
  gi.table_select = {0, 0, 0};
  if (a1 > 0)
    gi.table_select[0] = new_choose_table(ix, 0, a1);
  if (bigv > a1)
    gi.table_select[1] = new_choose_table(ix, a1, bigv);

  int bv = 0;
  if (gi.table_select[0]) bv += count_bit(ix, 0, a1, gi.table_select[0]);
  if (gi.table_select[1]) bv += count_bit(ix, a1, bigv, gi.table_select[1]);
  return c1 + bv;
}

// part2 (scalefactor) length for a short block (mixed_block_flag == 0):
// 3 windows × (6 bands at slen1 + 6 bands at slen2).
int short_part2_length(int compress) {
  const int slen1 = tables::slen1_tab[static_cast<std::size_t>(compress)];
  const int slen2 = tables::slen2_tab[static_cast<std::size_t>(compress)];
  return NUM_WINDOWS * (6 * slen1 + 6 * slen2);
}

// Bin-search the global gain so the short Huffman bit count is just under
// `target`. Mirrors quant::bin_search_StepSize.
int short_bin_search(int target, int* ix, GrInfo& gi, int scale,
                     const std::array<std::array<int, NUM_SFB_SHORT>, NUM_WINDOWS>& sf,
                     const std::array<int, NUM_WINDOWS>& sbg,
                     const std::array<Seg, tables::NUM_SCFB_SHORT * NUM_WINDOWS>& segs, int nseg,
                     EncoderState& s) {
  int next_val = -120, count = 120;
  while (count > 1) {
    const int half = count / 2;
    int bit;
    if (quantize_all(ix, next_val + half, scale, sf, sbg, segs, nseg, s) > 8192) {
      bit = 100000;
    } else {
      bit = short_huffman_bits(ix, gi);
    }
    if (bit < target) count = half;
    else { next_val += half; count -= half; }
  }
  return next_val;
}

// Pick the smallest-part2 scalefac_compress whose (slen1, slen2) can represent
// the scalefactors actually used. Returns -1 if none (caller falls back to 0).
int choose_compress(const std::array<std::array<int, NUM_SFB_SHORT>, NUM_WINDOWS>& sf) {
  int need1 = 0, need2 = 0;
  for (int w = 0; w < NUM_WINDOWS; ++w) {
    for (int sfb = 0; sfb < 6; ++sfb) need1 = std::max(need1, sf[static_cast<std::size_t>(w)][static_cast<std::size_t>(sfb)]);
    for (int sfb = 6; sfb < 12; ++sfb) need2 = std::max(need2, sf[static_cast<std::size_t>(w)][static_cast<std::size_t>(sfb)]);
  }
  int best = -1, best_p2 = 1 << 30;
  for (int c = 0; c < 16; ++c) {
    const int s1 = tables::slen1_tab[static_cast<std::size_t>(c)];
    const int s2 = tables::slen2_tab[static_cast<std::size_t>(c)];
    const int cap1 = (s1 > 0) ? ((1 << s1) - 1) : 0;
    const int cap2 = (s2 > 0) ? ((1 << s2) - 1) : 0;
    if (cap1 < need1 || cap2 < need2) continue;
    const int p2 = NUM_WINDOWS * (6 * s1 + 6 * s2);
    if (p2 < best_p2) { best_p2 = p2; best = c; }
  }
  return best;
}

} // namespace

bool short_block_quantize(int max_bits, int* ix, int gr, int ch, EncoderState& s) {
  auto& gi = s.side_info.gr[gr].ch[ch].tt;

  std::array<Seg, tables::NUM_SCFB_SHORT * NUM_WINDOWS> segs{};
  const int nseg = build_segments(s, segs);

  // Per-band scalefactors and per-window subblock_gain.
  std::array<std::array<int, NUM_SFB_SHORT>, NUM_WINDOWS> sf{};
  std::array<int, NUM_WINDOWS> sbg{0, 0, 0};
  const int scale = 0;

  // --- Per-window subblock_gain ---
  // A single global gain across all three short windows would be set by the
  // loudest window (e.g. the kick transient), quantizing the quiet windows
  // (a bass line sustaining through the attack) far too coarsely — audible as
  // bass quant noise at attacks, worst on loud material. subblock_gain gives
  // each window its own gain offset: quieter windows get a larger subblock_gain
  // (a finer effective step) so their low-level content keeps its precision
  // while the loud window stays coarse. We normalise to the loudest window by
  // its peak magnitude (one subblock_gain unit == 8 quarter-dB gain steps).
  {
    std::array<std::int32_t, NUM_WINDOWS> wpeak{0, 0, 0};
    for (int si = 0; si < nseg; ++si) {
      const Seg& sg = segs[static_cast<std::size_t>(si)];
      for (int i = sg.begin; i < sg.begin + sg.len; ++i)
        wpeak[static_cast<std::size_t>(sg.w)] =
            std::max(wpeak[static_cast<std::size_t>(sg.w)], s.xrabs[static_cast<std::size_t>(i)]);
    }
    std::int32_t gpeak = std::max({wpeak[0], wpeak[1], wpeak[2]});
    if (gpeak > 0) {
      for (int w = 0; w < NUM_WINDOWS; ++w) {
        if (wpeak[w] <= 0) { sbg[w] = 7; continue; } // silent window: max fineness
        const double ratio = static_cast<double>(gpeak) / static_cast<double>(wpeak[w]);
        const int steps = static_cast<int>(std::lround(std::log2(ratio) * 4.0));
        sbg[w] = std::clamp(steps / 8, 0, 7);
      }
    }
  }

  // Per-(w, sfb) signal energy (linear, MDCT² scale) for the noise loop.
  std::array<std::array<double, NUM_SFB_SHORT>, NUM_WINDOWS> e_band{};
  for (int si = 0; si < nseg; ++si) {
    const Seg& sg = segs[static_cast<std::size_t>(si)];
    if (sg.sfb >= NUM_SFB_SHORT) continue;
    double e = 0.0;
    for (int i = sg.begin; i < sg.begin + sg.len; ++i) {
      const double v = s.xr[i];
      e += v * v;
    }
    e_band[static_cast<std::size_t>(sg.w)][static_cast<std::size_t>(sg.sfb)] = e;
  }

  // Reserve part2 headroom assuming a modest compress; bin-search huffman gain.
  int compress = 0;
  int target_huff = std::max(0, max_bits - short_part2_length(compress));

  int gain = short_bin_search(target_huff, ix, gi, scale, sf, sbg, segs, nseg, s);
  int maxv = quantize_all(ix, gain, scale, sf, sbg, segs, nseg, s);
  if (maxv > 8192) { gain += 1; quantize_all(ix, gain, scale, sf, sbg, segs, nseg, s); }
  int huff = short_huffman_bits(ix, gi);

  // --- Per-band noise-shaping loop ---
  // Each iteration raises the scalefactor of the perceptually most-damaged band
  // and re-fits the global gain to budget. Bumping a band's sf by 1 and raising
  // the global gain by 1 to compensate makes that band one step finer while the
  // rest go one step coarser — i.e. it trades bits *into* the chosen band. We
  // steer that trade toward the low frequencies: during a loud attack a single
  // global gain (set by the kick) quantizes the quiet sustained bass to zero,
  // and an unweighted "worst NMR" rule spends the bits on the loud kick bands
  // instead. A strong low-frequency weight keeps the bass alive through the
  // attack, which is exactly where its quant noise is otherwise audible.
  constexpr int MAX_SHAPE_ITERS = 40;
  constexpr int SF_CAP = 15; // upper sf bound (compress selection enforces real cap)
  auto lf_weight = [](int sfb) {
    // Bass (sfb 0 ≈ 0–460 Hz) and low-mids weighted far above the rest.
    if (sfb <= 1) return 12.0;
    if (sfb <= 3) return 4.0;
    if (sfb <= 5) return 1.5;
    return 1.0;
  };
  for (int it = 0; it < MAX_SHAPE_ITERS; ++it) {
    // Find the band with the highest perceptually-weighted relative noise.
    int worst_w = -1, worst_sfb = -1;
    double worst_score = 0.30; // ignore bands already coded well (noise < 30% of energy, unweighted)
    for (int si = 0; si < nseg; ++si) {
      const Seg& sg = segs[static_cast<std::size_t>(si)];
      if (sg.sfb >= NUM_SFB_SHORT) continue;
      if (sf[static_cast<std::size_t>(sg.w)][static_cast<std::size_t>(sg.sfb)] >= SF_CAP) continue;
      const double eb = e_band[static_cast<std::size_t>(sg.w)][static_cast<std::size_t>(sg.sfb)];
      if (eb <= 0.0) continue;
      // Reconstruction noise for this band at current quant.
      const int g = eff_gain(gain, scale, sf[static_cast<std::size_t>(sg.w)][static_cast<std::size_t>(sg.sfb)], sbg[static_cast<std::size_t>(sg.w)]);
      const std::int32_t scalei = s.steptabi[static_cast<std::size_t>(g + 127)];
      const double dequant = (scalei > 0) ? 4294967296.0 / static_cast<double>(scalei) : 0.0;
      double noise = 0.0;
      for (int i = sg.begin; i < sg.begin + sg.len; ++i) {
        const double xhat = tables::pow43_tab[static_cast<std::size_t>(ix[i])] * dequant;
        const double diff = static_cast<double>(s.xrabs[static_cast<std::size_t>(i)]) - xhat;
        noise += diff * diff;
      }
      // Relative noise (1.0 == band fully lost) times the LF perceptual weight.
      const double score = (noise / eb) * lf_weight(sg.sfb);
      if (score > worst_score) { worst_score = score; worst_w = sg.w; worst_sfb = sg.sfb; }
    }
    if (worst_w < 0) break;

    // Tentatively bump that band's scalefactor and re-fit gain to budget.
    auto sf_trial = sf;
    sf_trial[static_cast<std::size_t>(worst_w)][static_cast<std::size_t>(worst_sfb)] += 1;
    const int c_trial = choose_compress(sf_trial);
    if (c_trial < 0) break;
    const int tgt = std::max(0, max_bits - short_part2_length(c_trial));
    const int g_trial = short_bin_search(tgt, ix, gi, scale, sf_trial, sbg, segs, nseg, s);
    int mv = quantize_all(ix, g_trial, scale, sf_trial, sbg, segs, nseg, s);
    if (mv > 8192) break;
    const int h_trial = short_huffman_bits(ix, gi);
    if (h_trial + short_part2_length(c_trial) > max_bits) {
      // Did not fit; revert and stop.
      quantize_all(ix, gain, scale, sf, sbg, segs, nseg, s);
      short_huffman_bits(ix, gi);
      break;
    }
    sf = sf_trial;
    compress = c_trial;
    gain = g_trial;
    huff = h_trial;
  }

  // Re-pick compress for the final sf set and re-quantize at the final gain to
  // leave ix / gi consistent with what will be emitted.
  {
    const int c = choose_compress(sf);
    compress = (c >= 0) ? c : 0;
    maxv = quantize_all(ix, gain, scale, sf, sbg, segs, nseg, s);
    if (maxv > 8192) { gain += 1; quantize_all(ix, gain, scale, sf, sbg, segs, nseg, s); }
    huff = short_huffman_bits(ix, gi);
  }

  // --- Hard budget enforcement ---
  // The bin search is approximate and the noise loop / compress reselection can
  // leave part2_3 above the granule budget. A short granule that overruns its
  // slot corrupts the fixed-size frame, so coarsen the global gain until it
  // fits (gain -> 127 drives every coefficient to zero, guaranteeing success).
  while (short_part2_length(compress) + huff > max_bits && gain < 127) {
    ++gain;
    quantize_all(ix, gain, scale, sf, sbg, segs, nseg, s);
    huff = short_huffman_bits(ix, gi);
  }

  // --- Black-spot guard: never let a granule with energy decode to silence. ---
  if (s.xrmax > 0) {
    bool all_zero = true;
    for (int i = 0; i < GRANULE_SAMPLES; ++i)
      if (ix[i] != 0) { all_zero = false; break; }
    if (all_zero) {
      int bin = 0;
      std::int32_t mx = 0;
      for (int i = 0; i < GRANULE_SAMPLES; ++i)
        if (s.xrabs[static_cast<std::size_t>(i)] > mx) { mx = s.xrabs[static_cast<std::size_t>(i)]; bin = i; }
      ix[bin] = 1;
      huff = short_huffman_bits(ix, gi);
    }
  }

  // --- Commit side info ---
  gi.window_switching_flag = 1;
  gi.block_type = BLOCK_SHORT;
  gi.mixed_block_flag = 0;
  gi.subblock_gain = sbg;
  gi.scalefac_scale = scale;
  gi.preflag = 0;
  gi.scalefac_compress = compress;
  gi.quantizerStepSize = gain;
  gi.global_gain = gain + 210;
  gi.part2_length = short_part2_length(compress);
  gi.part2_3_length = gi.part2_length + huff;

  // Write the per-window scalefactors (sfb 0..11; sfb 12 implicit 0).
  for (int w = 0; w < NUM_WINDOWS; ++w)
    for (int sfb = 0; sfb < NUM_SFB_SHORT; ++sfb)
      s.scalefactor.s[gr][ch][w][sfb] = sf[static_cast<std::size_t>(w)][static_cast<std::size_t>(sfb)];

  return true;
}

} // namespace rmps::quant
