#include "quant/iteration_loop.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include "core/reservoir.hpp"
#include "core/tables.hpp"
#include "post/hf_adaptor.hpp"
#include "post/lowpass.hpp"
#include "post/masking.hpp"
#include "post/ms_stereo.hpp"
#include "post/perceptual.hpp"
#include "quant/beam_search.hpp"
#include "quant/quantize.hpp"
#include "quant/short_block.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>

namespace rmps::quant {

namespace {

/// Zero MDCT coefficients above `cutoff_hz` for `(ch, gr)`. Applied per-granule
/// on top of the channel-wide LPF when the transient gate fires.
void apply_transient_cutoff(EncoderState &s, int ch, int gr, double cutoff_hz) {
  const double nyq = s.wave.samplerate * 0.5;
  if (cutoff_hz >= nyq - 1.0)
    return;
  auto &arr = s.mdct_freq[ch][gr];

  if (s.block_type[ch][gr] == BLOCK_SHORT) {
    // Short layout: coefficients are grouped (sfb, window, line). Each window
    // spans 192 frequency lines over [0, nyquist), so zero every line at or
    // above the cutoff across all three windows (transmission order).
    const double cutoff_line = cutoff_hz / nyq * 192.0;
    const auto &sb_s =
        tables::scale_fact_band_index_short[s.mpeg.samplerate_index];
    int ti = 0;
    for (int sfb = 0; sfb < tables::NUM_SCFB_SHORT - 1; ++sfb) {
      for (int w = 0; w < NUM_WINDOWS; ++w) {
        for (int f = sb_s[sfb]; f < sb_s[sfb + 1]; ++f) {
          if (static_cast<double>(f) >= cutoff_line)
            arr[ti] = 0;
          ++ti;
        }
      }
    }
    return;
  }

  int cutoff_bin =
      static_cast<int>(cutoff_hz * 576.0 * 2.0 / s.wave.samplerate);
  if (cutoff_bin < 0)
    cutoff_bin = 0;
  if (cutoff_bin >= GRANULE_SAMPLES)
    return;
  for (int i = cutoff_bin; i < GRANULE_SAMPLES; ++i)
    arr[i] = 0;
}

/// Zero out valleys between spectral peaks (aggressively prune noise floor
/// between peaks) taking harmonic/overtone relationships into account to
/// protect timbre.
/// The effect is adaptively scaled based on the target bitrate to optimize the bit budget.
void apply_peak_valley_zeroing(EncoderState &s) {
  // Compute bitrate-dependent scale factor:
  // 128 kbps -> 1.0x, 320 kbps -> 0.4x (gentle), 32 kbps -> 2.0x (aggressive)
  double bitrate_scale = 1.0;
  const int bitr = s.mpeg.bitr;
  if (bitr > 0) {
    if (bitr >= 128) {
      double t = static_cast<double>(bitr - 128) / (320.0 - 128.0);
      bitrate_scale = 1.0 - 0.6 * t;
    } else {
      double t = static_cast<double>(128 - bitr) / (128.0 - 32.0);
      t = std::clamp(t, 0.0, 1.0);
      bitrate_scale = 1.0 + 1.0 * t;
    }
  }

  for (int ch = 0; ch < s.wave.channels; ++ch) {
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
      // Short blocks store coefficients in a reordered (sfb, window, line)
      // layout; the harmonic/peak analysis below assumes the linear long
      // layout, so skip it for short granules.
      if (s.block_type[ch][gr] == BLOCK_SHORT)
        continue;
      auto &arr = s.mdct_freq[ch][gr];

      // 1. Compute magnitudes
      std::array<double, GRANULE_SAMPLES> mag{};
      for (int i = 0; i < GRANULE_SAMPLES; ++i) {
        mag[i] = std::abs(static_cast<double>(arr[i]));
      }

      // 2. Find fundamental frequency and identify its harmonics (2nd to 8th
      // overtones)
      int fund_bin = -1;
      double max_fund_val = 0.0;
      for (int i = 4; i < 120; ++i) {
        if (mag[i] > max_fund_val) {
          max_fund_val = mag[i];
          fund_bin = i;
        }
      }

      std::array<bool, GRANULE_SAMPLES> is_harmonic{};
      if (fund_bin >= 0 && max_fund_val > 1000.0) {
        for (int k = 2; k <= 128; ++k) {
          int harm_bin = k * fund_bin;
          int start = std::max(0, harm_bin - 2);
          int end = std::min(GRANULE_SAMPLES, harm_bin + 3);
          for (int j = start; j < end; ++j) {
            is_harmonic[j] = true;
          }
        }
      }

      // 3. Compute local maximums using a frequency-dependent window size
      std::array<double, GRANULE_SAMPLES> local_max{};
      for (int i = 0; i < GRANULE_SAMPLES; ++i) {
        int w = 8;
        if (i >= 250)
          w = 24;
        else if (i >= 100)
          w = 16;

        // Scale window size based on bitrate: wider at lower bitrates
        w = static_cast<int>(static_cast<double>(w) * std::sqrt(bitrate_scale) + 0.5);
        w = std::clamp(w, 3, 48);

        double mx = 0.0;
        int start = std::max(0, i - w);
        int end = std::min(GRANULE_SAMPLES, i + w + 1);
        for (int j = start; j < end; ++j) {
          if (mag[j] > mx) {
            mx = mag[j];
          }
        }
        local_max[i] = mx;
      }

      // 4. Zero out valleys between peaks (above 1 kHz, i.e., bin 26)
      for (int i = 26; i < GRANULE_SAMPLES; ++i) {
        double threshold = 0.06;
        if (i >= 250)
          threshold = 0.20;
        else if (i >= 100)
          threshold = 0.12;

        // Apply bitrate scale factor and clamp to a safe maximum
        threshold *= bitrate_scale;
        threshold = std::min(threshold, 0.40);

        // If the bin is near a harmonic/overtone, lower the zeroing threshold
        // to preserve the overtone structure (timbre).
        if (is_harmonic[i]) {
          threshold *= 0.2;
        }

        if (mag[i] < local_max[i] * threshold) {
          arr[i] = 0;
        }
      }
    }
  }
}

} // namespace

void iteration_loop(EncoderState &s) {
  // ----- Per-frame pre-quantization processing -----
  post::apply_ms_stereo(s);
  s.hf_adaptor_apply();
  post::apply_lowpass(s);
  apply_peak_valley_zeroing(s);

  // Per-granule transient cutoff (kicks pre-echo mitigation): applied to
  // every channel of any granule the gate flagged.
  for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
    const double cutoff = s.transient_cutoff_hz[gr];
    if (cutoff < s.wave.samplerate * 0.5 - 1.0) {
      for (int ch = 0; ch < s.wave.channels; ++ch) {
        // M channel (ch 0) keeps a slightly wider band than S so the
        // attack's body still has spectral content; S is trimmed a little
        // more (the stereo side is less exposed during a strike) but not
        // slammed — an over-narrow Side collapses the image at every hit.
        const double ch_cutoff = (ch == 0 && s.wave.channels == 2)
                                     ? std::min(cutoff * 1.4, 22050.0)
                                     : std::min(cutoff * 1.15, 22050.0);
        apply_transient_cutoff(s, ch, gr, ch_cutoff);
      }
    }
  }

  // Refresh masking model parameters and compute per-(ch, gr) thresholds
  // BEFORE beam_search sees this granule. Note: history advances in
  // (gr, ch) order so post-mask is causal frame-by-frame.
  s.mask_params = post::make_masking_params(s);
  for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
    for (int ch = 0; ch < s.wave.channels; ++ch) {
      // Short granules store a reordered (sfb, window, line) spectrum; feeding
      // it to the long-layout masking model would compute garbage thresholds
      // AND poison history.prev_thr, corrupting the post-mask carry into the
      // following granule (audible as bass quant noise right after an attack).
      // The short quantizer uses its own per-band energy model, so skip the
      // long model here and leave the masking history untouched across the
      // short granule.
      if (s.block_type[ch][gr] == BLOCK_SHORT)
        continue;
      post::compute_masking_thresholds(s, ch, gr, s.mask_params,
                                       s.mask_history[ch],
                                       s.mask_thresholds[ch][gr]);
    }
  }

  // --- Masking-based Mid/Side channel band disabling & budget reallocation ---
  int ms_xfer[MAX_GRANULES] = {0};
  if (s.wave.channels == 2 && (s.mpeg.mode_ext & 2) && s.mpeg.bitr <= 128) {
    const auto &band_table =
        tables::scale_fact_band_index[s.mpeg.samplerate_index];
    const int side_base = s.mean_bits / s.wave.channels;
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
      // Short granules use the reordered layout; the long-sfb band pruning
      // below does not map onto it, so leave them untouched.
      if (s.block_type[0][gr] == BLOCK_SHORT)
        continue;
      const auto &mt_mid = s.mask_thresholds[0][gr];
      const auto &mt_side = s.mask_thresholds[1][gr];
      auto &mdct_mid = s.mdct_freq[0][gr];
      auto &mdct_side = s.mdct_freq[1][gr];

      int pruned_width_side = 0;
      int pruned_width_mid = 0;
      // sfb 0 (the lowest band, DC–~170 Hz) is never pruned in either
      // channel: zeroing the fundamental is audible as a bass dropout no
      // matter what the masking model claims about it.
      for (int sfb = 1; sfb < NUM_SFB_LONG; ++sfb) {
        const double e_s = mt_side.e_band[sfb];
        const double e_m = mt_mid.e_band[sfb];
        const double thr_s = mt_side.thr[sfb];
        const double thr_m = mt_mid.thr[sfb];

        // Asymmetric pruning. Mid carries the mono/dominant content and is
        // pruned on the original (loose) rule. Side carries the stereo
        // image: zeroing a Side band collapses the image there, which is
        // audible on wide material even when the band sits below its own
        // masking threshold. So Side is pruned only when it is genuinely
        // negligible — below its own threshold AND tiny relative to Mid —
        // not merely below the (often generous) Mid threshold.
        const bool prune_mid_cand =
            (e_m <= thr_m * 0.5) || (e_m <= thr_s * 0.5) || (e_m < 0.01 * e_s);
        const bool prune_side_cand =
            (e_s <= thr_s * 0.35) && (e_s < 0.004 * e_m);

        bool prune_mid = false;
        bool prune_side = false;

        if (prune_mid_cand && prune_side_cand) {
          // Prevent zeroing out both: keep the one with higher energy
          if (e_m >= e_s) {
            prune_side = true;
          } else {
            prune_mid = true;
          }
        } else if (prune_mid_cand) {
          prune_mid = true;
        } else if (prune_side_cand) {
          prune_side = true;
        }

        const int start = band_table[sfb];
        const int end = band_table[sfb + 1];
        const int width = end - start;

        if (prune_mid) {
          for (int i = start; i < end; ++i) {
            mdct_mid[i] = 0;
          }
          pruned_width_mid += width;
        } else if (prune_side) {
          for (int i = start; i < end; ++i) {
            mdct_side[i] = 0;
          }
          pruned_width_side += width;
        }
      }

      // Reallocate budget symmetrically:
      // Net transfer fraction (Side-to-Mid is positive, Mid-to-Side is
      // negative)
      const double f_side = static_cast<double>(pruned_width_side) /
                            static_cast<double>(GRANULE_SAMPLES);
      const double f_mid = static_cast<double>(pruned_width_mid) /
                           static_cast<double>(GRANULE_SAMPLES);
      const double f_net = f_side - f_mid;

      constexpr double MAX_TRANSFER_FRAC = 1.0 - 0.35; // 0.65
      double transfer_frac = f_net * 0.8;
      transfer_frac =
          std::clamp(transfer_frac, -MAX_TRANSFER_FRAC, MAX_TRANSFER_FRAC);
      ms_xfer[gr] = static_cast<int>(transfer_frac * side_base);
    }
  }

  int gr0_sf_per_ch[MAX_CHANNELS][NUM_SFB_LONG];
  bool gr0_sf_valid[MAX_CHANNELS] = {false, false};

  // Aggregate stats for FrameStats.
  double raw_signal_total = 0.0;
  double raw_dist_total = 0.0;
  double w_signal_total = 0.0;
  double w_dist_total = 0.0;
  int granule_count = 0;
  int granule_bits_sum = 0;
  int granule_budget_sum = 0;
  int hf_rdo_total = 0;
  s.frame_outer_loop_fallbacks = 0;
  s.frame_granule_stats.clear();

  // --- Intra-frame bit pool -------------------------------------------
  // Frames are emitted self-contained (main_data_begin = 0), so any bits a
  // granule leaves on the table become pure stuffing at frame_end. Instead
  // of letting surplus evaporate, carry it forward to the remaining
  // (ch, gr) slots of the SAME frame: granules that need fewer bits than
  // their flat share (quiet, sparse, or post-RD-split savings) donate to
  // the ones encoded after them. Total frame size is unchanged — this only
  // converts would-be stuffing into coded precision.
  int frame_pool_bits = 0;

  // --- Attack surge ---------------------------------------------------
  // Pre-echo scales directly with the attack granule's quantization step:
  // measured on silence→burst material, 128k→320k lowers the decoded
  // pre-attack noise floor by ~10 dB, dwarfing every window-domain trick.
  // So shift budget *within the frame* toward the granule that carries
  // the attack, funded by its sibling. The donation is graded by how
  // expendable the sibling is: a silent or pre-window granule (quiet by
  // definition — it precedes the strike) gives up half its share, a
  // normal granule a quarter. Total frame bits are unchanged, so this is
  // CBR-safe; the ABR transient hint stacks on top for ABR mode.
  double gr_bias[MAX_GRANULES] = {1.0, 1.0};
  if (s.mpeg.granules_per_frame == 2) {
    // A sub-window-2 attack at the end of the previous frame lands (after
    // the polyphase delay) inside THIS frame's gr0 MDCT window — treat
    // gr0 as attack-bearing so its quantization step, which decodes into
    // the pre-attack region, gets the surge too.
    const bool spill0 =
        s.prev_frame_attack && s.prev_frame_attack_sub >= 2;
    const auto has_attack = [&](int g) {
      return s.transient_flags[g] || s.attack_from_silence_flags[g]
          || (g == 0 && spill0);
    };
    for (int g = 0; g < 2; ++g) {
      const int sib = 1 - g;
      if (has_attack(g) && !has_attack(sib)) {
        const bool sib_quiet =
            s.silent_flags[sib] || s.pre_window_flags[sib];
        const double surge = sib_quiet ? 0.50 : 0.25;
        gr_bias[g]   = 1.0 + surge;
        gr_bias[sib] = 1.0 - surge;
      }
    }
  }

  // --- Cross-frame reservoir draw --------------------------------------
  // Quiet frames bank their slack (reservoir::frame_end keeps up to
  // ResvMax un-stuffed; l3_bitstream realises it via main_data_begin).
  // Spend the bank where it buys the most: granules carrying an attack.
  // Draw at most 70% per frame so consecutive attacks each get a share,
  // and never plan below zero — reservoir::adjust subtracts actual spend.
  int resv_draw_per_slot = 0;
  {
    int n_att_slots = 0;
    if (s.mpeg.granules_per_frame == 2) {
      const bool spill0 =
          s.prev_frame_attack && s.prev_frame_attack_sub >= 2;
      for (int g = 0; g < 2; ++g)
        if (s.transient_flags[g] || s.attack_from_silence_flags[g]
            || (g == 0 && spill0))
          n_att_slots += s.wave.channels;
    }
    // Only draw against the bitrate-tapered bank: below 96 kbps the bank is
    // shrunk toward zero (reservoir::effective_resv_max) to stop the bass
    // pumping, so the attack draw must respect the same limit.
    const int eff_bank =
        std::min(s.ResvSize, reservoir::effective_resv_max(s));
    if (n_att_slots > 0 && eff_bank > 0)
      resv_draw_per_slot = ((eff_bank * 7) / 10) / n_att_slots;
  }

  // --- Reservoir-pressure λ coupling ------------------------------------
  // The beam's cost is perceptual + λ_R·bits, so λ is the natural knob to
  // integrate the reservoir state into the search itself: an attack
  // granule searches with a lower λ (bits are cheap — the bank pays),
  // while a lean bank raises λ everywhere so quiet frames re-fill it.
  const int eff_resv_max = reservoir::effective_resv_max(s);
  const double resv_fill = (eff_resv_max > 0)
      ? std::clamp(static_cast<double>(s.ResvSize) / eff_resv_max, 0.0, 1.0)
      : 1.0;  // no bank to refill (low rate) → no λ penalty

  for (int ch = 0; ch < s.wave.channels; ++ch) {
    for (auto &v : s.side_info.scfsi[ch])
      v = 0;
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
      auto *ix = s.l3_enc[ch][gr].data();
      s.xr = s.mdct_freq[ch][gr].data();

      s.xrsq.fill(0);
      s.xrabs.fill(0);
      s.xrmax = 0;
      for (int i = 0; i < GRANULE_SAMPLES; ++i) {
        const std::int32_t v = s.xr[i];
        const std::int64_t sq = static_cast<std::int64_t>(v) * v +
                                static_cast<std::int64_t>(0x40000000LL);
        s.xrsq[i] = fx::i32(sq >> 31);
        const std::int32_t a = std::abs(v);
        s.xrabs[i] = a;
        if (a > s.xrmax)
          s.xrmax = a;
      }

      auto &gi = s.side_info.gr[gr].ch[ch].tt;
      gi.sfb_lmax = NUM_SFB_LONG;
      // ABR carries the silence/transient signal via the per-frame
      // bitrate index now — mean_bits is already small for silent
      // frames and large for attack frames. No further per-granule
      // budget shaping needed here.
      int max_bits = reservoir::max_reservoir_bits(s.pe[ch][gr], s);
      max_bits = std::clamp(
          static_cast<int>(max_bits * gr_bias[gr]), 0, 4095);
      const bool slot_has_attack =
          s.transient_flags[gr] || s.attack_from_silence_flags[gr]
          || (gr == 0 && s.prev_frame_attack && s.prev_frame_attack_sub >= 2);
      if (slot_has_attack && resv_draw_per_slot > 0)
        max_bits = std::min(max_bits + resv_draw_per_slot, 4095);
      // Reservoir-pressure λ for this granule's search: attack granules
      // buy bits cheap; a lean bank makes everyone frugal so it refills.
      s.lambda_granule_scale =
          slot_has_attack ? 0.6 : 1.0 + 0.3 * (1.0 - resv_fill);
      if (ms_xfer[gr] > 0) {
        if (ch == 0)
          max_bits += ms_xfer[gr]; // Mid gains
        else if (ch == 1)
          max_bits -= ms_xfer[gr]; // Side yields
        max_bits = std::clamp(max_bits, 0, 4095);
      }
      // Hand the accumulated intra-frame surplus to this granule. The
      // 4095 cap is the part2_3_length field limit; anything the cap
      // rejects stays in the pool for the next slot.
      const int base_grant = max_bits;
      max_bits = std::min(base_grant + frame_pool_bits, 4095);
      frame_pool_bits -= (max_bits - base_grant);

      for (auto &v : s.scalefactor.l[gr][ch])
        v = 0;
      for (auto &v : s.scalefactor.s[gr][ch])
        v.fill(0);
      for (auto &v : gi.slen)
        v = 0;
      gi.part2_3_length = 0;
      gi.big_values = 0;
      gi.count1 = 0;
      gi.scalefac_compress = 0;
      gi.table_select = {0, 0, 0};
      gi.region0_count = 0;
      gi.region1_count = 0;
      gi.part2_length = 0;
      gi.preflag = 0;
      gi.scalefac_scale = 0;
      gi.count1table_select = 0;
      gi.window_switching_flag = 0;
      gi.block_type = BLOCK_LONG;
      gi.mixed_block_flag = 0;
      gi.subblock_gain = {0, 0, 0};

      const bool is_short = (s.block_type[ch][gr] == BLOCK_SHORT);

      if (is_short && s.xrmax > 0) {
        // ----- Short-block quantization path -----
        short_block_quantize(max_bits, ix, gr, ch, s);
        granule_bits_sum += gi.part2_3_length;
        granule_budget_sum += max_bits;
        ++granule_count;
      } else if (s.xrmax > 0) {
        // For start/stop transition blocks the spectrum is still a 576-line
        // long layout, so the long beam_search path applies; only block_type /
        // window_switching differ in the side info, set below.
        // scfsi scalefactor sharing is only valid between two normal long
        // granules; disable it whenever window switching is involved.
        const int *gr0_ptr =
            (gr == 1 && gr0_sf_valid[ch] && s.block_type[ch][gr] == BLOCK_LONG)
                ? gr0_sf_per_ch[ch]
                : nullptr;
        // Start/stop transition blocks re-split their big-values region with
        // the fixed window-switching rule afterwards, which costs a few dozen
        // bits more than the beam's optimal split. Reserve headroom so that
        // recount fits within budget — otherwise the coarsening fallback below
        // discards the beam's (bass-protecting) scalefactors and the low end
        // turns to quant noise right at the attack.
        const int bt_g = s.block_type[ch][gr];
        const int beam_budget =
            (bt_g == BLOCK_START || bt_g == BLOCK_STOP)
                ? std::max(0, max_bits - 128)
                : max_bits;
        BeamResult br = beam_search(beam_budget, ix, gr, ch, s, gr0_ptr);
        if (!br.ok) {
          ++s.frame_outer_loop_fallbacks;
          for (auto &v : s.side_info.scfsi[ch])
            v = 0;
          gi.scalefac_compress = 0;
          gi.scalefac_scale = 0;
          gi.preflag = 0;
          for (int sfb = 0; sfb < NUM_SFB_LONG + 1; ++sfb)
            s.scalefactor.l[gr][ch][sfb] = 0;
          gi.part2_3_length = outer_loop(max_bits, ix, gr, ch, s);

          // Black-spot guard for the legacy fallback path. beam_search
          // protects its own output, but outer_loop can coarsen a
          // granule (Mid included) all the way to silence under
          // starvation, which decodes as a dropout. If the source
          // granule has energy but every coefficient went to zero,
          // restore the single most audible bin (a few count1 bits the
          // reservoir absorbs) and refresh the side-info fields.
          {
            bool all_zero = true;
            for (int i = 0; i < GRANULE_SAMPLES; ++i)
              if (ix[i] != 0) {
                all_zero = false;
                break;
              }
            if (all_zero) {
              int bin = 0;
              std::int32_t mx = 0;
              for (int i = 0; i < GRANULE_SAMPLES; ++i)
                if (s.xrabs[i] > mx) {
                  mx = s.xrabs[i];
                  bin = i;
                }
              ix[bin] = 1;
              calc_runlen(ix, gi);
              int bits = count1_bitcount(ix, gi);
              bits += best_subdivide_rd(ix, gi, s);
              gi.part2_3_length = gi.part2_length + bits;
            }
          }
        } else {
          granule_bits_sum += gi.part2_3_length;
          granule_budget_sum += max_bits;
          raw_signal_total += br.raw_signal_pow;
          raw_dist_total += br.raw_dist;
          w_signal_total += br.weighted_signal_pow;
          w_dist_total += br.cum_dist;
          hf_rdo_total += br.hf_rdo_bits_saved;
          ++granule_count;

          GranuleStats gs{};
          gs.gr = gr;
          gs.ch = ch;
          gs.gain = gi.quantizerStepSize;
          gs.preflag = gi.preflag;
          gs.scale = gi.scalefac_scale;
          gs.compress = gi.scalefac_compress;
          for (int b = 0; b < 4; ++b)
            gs.scfsi_pat[b] = s.side_info.scfsi[ch][b];
          gs.bits = gi.part2_3_length;
          gs.budget = max_bits;
          if (br.raw_signal_pow > 1.0 && br.raw_dist > 1.0)
            gs.raw_sdr_db = 10.0 * std::log10(br.raw_signal_pow / br.raw_dist);
          if (br.weighted_signal_pow > 1.0 && br.cum_dist > 1.0)
            gs.perc_sdr_db =
                10.0 * std::log10(br.weighted_signal_pow / br.cum_dist);
          gs.hf_rdo_bits_saved = br.hf_rdo_bits_saved;
          s.frame_granule_stats.push_back(gs);
        }
        if (gr == 0 && s.block_type[ch][0] == BLOCK_LONG) {
          for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb)
            gr0_sf_per_ch[ch][sfb] = s.scalefactor.l[0][ch][sfb];
          gr0_sf_valid[ch] = true;
        } else if (gr == 0) {
          gr0_sf_valid[ch] = false;
        }

        // Start/stop transition blocks: long-layout spectrum but window-
        // switched side info. The decoder forces region0_count=7,
        // region1_count=13 (region1Start = long sfb[8], region2Start = 576)
        // for window-switched non-short blocks, so re-split the big-values
        // region accordingly and recount, overriding the beam's optimal split.
        const int bt_now = s.block_type[ch][gr];
        if (bt_now == BLOCK_START || bt_now == BLOCK_STOP) {
          const auto &sfbl =
              tables::scale_fact_band_index[s.mpeg.samplerate_index];
          auto recount_forced = [&]() {
            const int bigv = gi.big_values << 1;
            const int r1 = std::min(sfbl[8], bigv);
            gi.region0_count = 7;
            gi.region1_count = 13;
            gi.address1 = r1;
            gi.address2 = bigv;
            gi.address3 = bigv;
            gi.table_select = {0, 0, 0};
            if (r1 > 0) gi.table_select[0] = new_choose_table(ix, 0, r1);
            if (bigv > r1) gi.table_select[1] = new_choose_table(ix, r1, bigv);
            const int c1 = count1_bitcount(ix, gi);
            int bv = 0;
            if (gi.table_select[0]) bv += count_bit(ix, 0, r1, gi.table_select[0]);
            if (gi.table_select[1]) bv += count_bit(ix, r1, bigv, gi.table_select[1]);
            return gi.part2_length + c1 + bv;
          };
          gi.part2_3_length = recount_forced();

          // The forced region split is less efficient than the beam's optimal
          // split, so it can push part2_3 a few dozen bits over the granule
          // budget. A transition block that overruns its slot corrupts the
          // fixed-size frame, so coarsen to budget: drop the scalefactors
          // (scalefac_compress 0 -> part2 = 0) and escalate the global gain,
          // recounting with the forced regions, until it fits.
          if (gi.part2_3_length > max_bits) {
            for (auto &v : s.scalefactor.l[gr][ch]) v = 0;
            gi.scalefac_compress = 0;
            gi.scalefac_scale = 0;
            gi.preflag = 0;
            gi.part2_length = 0;
            int step = gi.quantizerStepSize;
            for (;;) {
              const int mv = quantize(ix, step, s);
              if (mv <= 8192) {
                calc_runlen(ix, gi);
                const int p = recount_forced();
                if (p <= max_bits) { gi.part2_3_length = p; break; }
              }
              if (step >= 127) {
                std::fill_n(ix, GRANULE_SAMPLES, 0);
                calc_runlen(ix, gi);
                gi.part2_3_length = recount_forced();
                break;
              }
              ++step;
            }
            gi.quantizerStepSize = step;
          }
        }
      }

      // Window-switching side-info flags. Set unconditionally from the decided
      // block type — even for an all-silent (xrmax == 0) start/stop/short
      // granule that ran no quantizer — otherwise it would emit as a long
      // block and break the decoder's required window-transition sequence
      // (e.g. an attack-from-silence START whose late onset the start window
      // attenuates to zero). The short path and start/stop override set the
      // same values when they run; this is the catch-all.
      if (s.block_type[ch][gr] != BLOCK_LONG) {
        gi.window_switching_flag = 1;
        gi.block_type = s.block_type[ch][gr];
        gi.mixed_block_flag = 0;
      }

      // Whatever this granule did not use flows back into the pool
      // (clamped at 0: the black-spot guard may exceed the advisory
      // budget by a few count1 bits, which the frame slack absorbs).
      frame_pool_bits += std::max(0, max_bits - gi.part2_3_length);
      reservoir::adjust(gi, s);
      gi.global_gain = gi.quantizerStepSize + 210;
    }
  }

  // Publish frame totals on s.
  s.frame_raw_signal_pow = raw_signal_total;
  s.frame_raw_dist = raw_dist_total;
  s.frame_w_signal_pow = w_signal_total;
  s.frame_w_dist = w_dist_total;
  s.frame_granule_count = granule_count;
  s.frame_total_bits = granule_bits_sum;
  s.frame_total_budget = granule_budget_sum;
  s.frame_hf_rdo_saved = hf_rdo_total;

  reservoir::frame_end(s);
}

} // namespace rmps::quant
