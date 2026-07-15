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
      if (fund_bin >= 0 && max_fund_val > 250.0) {  // 1000/4: headroom
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

double compute_adaptive_clamp(const EncoderState &s) {
  const int bitr = s.mpeg.bitr;
  double base_freq = 14000.0;
  if (bitr >= 192) {
    base_freq = 4000.0;
  } else if (bitr >= 32) {
    double t = static_cast<double>(bitr - 32) / (192.0 - 32.0);
    base_freq = 14000.0 - 10000.0 * t;
  }

  // Analyze spectrogram MDCT energy
  double e_lf = 0.0;
  double e_hf = 0.0;
  const double nyq = s.wave.samplerate * 0.5;
  const int bin_8k = static_cast<int>(8000.0 / nyq * 576.0);

  for (int ch = 0; ch < s.wave.channels; ++ch) {
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
      const auto &arr = s.mdct_freq[ch][gr];
      for (int i = 0; i < bin_8k; ++i) {
        double v = std::abs(static_cast<double>(arr[i]));
        e_lf += v * v;
      }
      for (int i = bin_8k; i < GRANULE_SAMPLES; ++i) {
        double v = std::abs(static_cast<double>(arr[i]));
        e_hf += v * v;
      }
    }
  }

  double hf_ratio = 0.0;
  if (e_lf > 0.0) {
    hf_ratio = e_hf / e_lf;
  }

  double adj_factor = 1.0;
  if (hf_ratio > 0.0) {
    // Map hf_ratio [0.001, 0.02] to adjustment factor [1.5, 0.6]
    // A higher hf_ratio means high HF energy, so we lower the clamp (more boost/protection).
    double t = (hf_ratio - 0.001) / (0.02 - 0.001);
    t = std::clamp(t, 0.0, 1.0);
    adj_factor = 1.5 - 0.9 * t;
  }

  double clamp_hz = base_freq * adj_factor;
  return std::clamp(clamp_hz, 4000.0, 14000.0);
}

} // namespace

inline int eff_gain(int gain, int scale, int sfv, int sbg) {
  return gain - 2 * (1 + scale) * sfv - (sbg << 3);
}

double calculate_bass_dist(int gr, int ch, const int* ix, const EncoderState& s) {
    return 0.0;
}

void iteration_loop(EncoderState &s) {
  // ----- Compute adaptive clamp frequency -----
  s.frame_clamp_hz = compute_adaptive_clamp(s);

  // ----- Per-frame pre-quantization processing -----
  post::apply_ms_stereo(s);

  // ----- Intensity-stereo candidate decision (Phase 1) -----
  // MS+IS (mode_ext = 3) frames: MPEG-1, long-block-only frames at <= 96
  // kbps, where the side LPF already zeroes ch1 above ~13 kHz so the
  // decoder-implicit intensity region lines up with sfb 20 (13.1 kHz+).
  // The L/R band levels the positions need must be captured NOW — the side
  // LPF below destroys the ch1 spectrum they are derived from.
  s.intensity_frame = false;
  double is_aL[MAX_GRANULES][NUM_SFB_LONG] = {};
  double is_aR[MAX_GRANULES][NUM_SFB_LONG] = {};
  double is_co[MAX_GRANULES][NUM_SFB_LONG] = {};
  if (s.mpeg.version == 3 && s.wave.channels == 2 &&
      (s.mpeg.mode_ext & 2) && s.mpeg.bitr <= 96) {
    bool all_long = true;
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr)
      if (s.block_type[0][gr] != BLOCK_LONG)
        all_long = false;
    if (all_long) {
      const auto &bt = tables::scale_fact_band_index[s.mpeg.samplerate_index];
      constexpr double INV_SQRT2 = 0.7071067811865475;
      for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
        const auto &M = s.mdct_freq[0][gr];
        const auto &T = s.mdct_freq[1][gr];
        for (int sfb = 11; sfb < NUM_SFB_LONG; ++sfb) {
          double eL = 0.0, eR = 0.0, cr = 0.0;
          for (int i = bt[sfb]; i < bt[sfb + 1]; ++i) {
            const double l = (static_cast<double>(M[i]) + T[i]) * INV_SQRT2;
            const double r = (static_cast<double>(M[i]) - T[i]) * INV_SQRT2;
            eL += l * l;
            eR += r * r;
            cr += l * r;
          }
          is_aL[gr][sfb] = std::sqrt(eL);
          is_aR[gr][sfb] = std::sqrt(eR);
          is_co[gr][sfb] = cr / (is_aL[gr][sfb] * is_aR[gr][sfb] + 1e-9);
        }
      }
      // Only frames with an actual off-center coherent pan in the intensity
      // band become candidates. Everything else keeps the plain-MS coding
      // path untouched — in particular ch1's HF noise-fill, which supplies
      // useful stereo "air" precisely where there is no clean pan to encode.
      for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
        const double aL = is_aL[gr][20];
        const double aR = is_aR[gr][20];
        if (aL + aR > 1.0 && is_co[gr][20] > 0.4) {
          const int pos = std::clamp(
              static_cast<int>(std::lround(std::atan2(aL, aR) *
                                           (12.0 / 3.141592653589793))),
              0, 6);
          if (pos != 3)
            s.intensity_frame = true;
        }
      }
    }
  }

  s.hf_adaptor_apply();
  post::apply_lowpass(s);
  post::apply_hf_gain_boost(s);
  apply_peak_valley_zeroing(s);

  // Per-granule transient cutoff (kicks pre-echo mitigation): applied to
  // every channel of any granule the gate flagged.
  for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
    const double cutoff = s.transient_cutoff_hz[gr];
    if (cutoff < s.wave.samplerate * 0.5 - 1.0) {
      for (int ch = 0; ch < s.wave.channels; ++ch) {
        // M channel (ch 0) keeps a slightly wider band than S so the
        // attack's body still has spectral content; S is killed more
        // aggressively (stereo side is least audible during a strike).
        const double ch_cutoff = (ch == 0 && s.wave.channels == 2)
                                     ? std::min(cutoff * 1.4, 22050.0)
                                     : cutoff;
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

      int pruned_bins_side = 0;
      int pruned_bins_mid = 0;
      const double nyq = s.wave.samplerate * 0.5;
      const int bin_clamp = static_cast<int>(s.frame_clamp_hz / nyq * 576.0);

      for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const double e_s = mt_side.e_band[sfb];
        const double e_m = mt_mid.e_band[sfb];
        const double thr_s = mt_side.thr[sfb];
        const double thr_m = mt_mid.thr[sfb];

        const int start = band_table[sfb];
        const int end = band_table[sfb + 1];
        const int width = end - start;

        // Masking thresholds scaled down to bin level
        const double thr_m_bin = thr_m / width;
        const double thr_s_bin = thr_s / width;

        for (int i = start; i < end; ++i) {
          const double m_val = std::abs(static_cast<double>(mdct_mid[i]));
          const double s_val = std::abs(static_cast<double>(mdct_side[i]));
          const double e_m_bin = m_val * m_val;
          const double e_s_bin = s_val * s_val;

          // Symmetric candidates for pruning at bin level
          bool prune_mid_cand =
              (e_m_bin <= thr_m_bin * 0.5) || (e_m_bin <= thr_s_bin * 0.5) || (e_m_bin < 0.02 * e_s_bin);
          bool prune_side_cand =
              (e_s_bin <= thr_s_bin * 0.5) || (e_s_bin <= thr_m_bin * 0.5) || (e_s_bin < 0.02 * e_m_bin);

          // Below clamp protection for Mid channel: make it extremely hard to prune Mid
          if (i < bin_clamp) {
            prune_mid_cand = (e_m_bin <= thr_m_bin * 0.02) && (e_m_bin < 0.001 * e_s_bin);
          }

          bool prune_mid = false;
          bool prune_side = false;

          if (prune_mid_cand && prune_side_cand) {
            // Protect the larger energy bin from being zeroed out.
            // If below clamp, protect Mid and prune Side instead.
            if (i < bin_clamp) {
              prune_side = true;
            } else {
              if (e_m_bin >= e_s_bin) {
                prune_side = true;
              } else {
                prune_mid = true;
              }
            }
          } else if (prune_mid_cand) {
            prune_mid = true;
          } else if (prune_side_cand) {
            prune_side = true;
          }

          if (prune_mid) {
            mdct_mid[i] = 0;
            pruned_bins_mid++;
          } else if (prune_side) {
            mdct_side[i] = 0;
            pruned_bins_side++;
          }
        }
      }

      // Reallocate budget symmetrically based on bin counts:
      // Net transfer fraction (Side-to-Mid is positive, Mid-to-Side is negative)
      const double f_side = static_cast<double>(pruned_bins_side) /
                            static_cast<double>(GRANULE_SAMPLES);
      const double f_mid = static_cast<double>(pruned_bins_mid) /
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
  double bass_dist_total = 0.0;

  // Reservoir snapshot for exact end-of-frame accounting (reservoir.cpp
  // frame_end) and for the physical frame-capacity cap below. adjust()
  // mutates ResvSize granule by granule as the planner's running view.
  s.resv_frame_start = s.ResvSize;
  int frame_used_bits = 0;

  // --- Intra-frame bit pool -------------------------------------------
  // Frames are emitted self-contained (main_data_begin = 0), so any bits a
  // granule leaves on the table become pure stuffing at frame_end. Instead
  // of letting surplus evaporate, carry it forward to the remaining
  // (ch, gr) slots of the SAME frame: granules that need fewer bits than
  // their flat share (quiet, sparse, or post-RD-split savings) donate to
  // the ones encoded after them. Total frame size is unchanged — this only
  // converts would-be stuffing into coded precision.
  int frame_pool_bits = 0;

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
      // Physical frame-capacity cap: a granule must never commit bits the
      // frame plus the reservoir cannot hold — a deficit would need a
      // negative main_data_begin, i.e. an invalid stream. The 64-bit margin
      // absorbs the black-spot guard's small budget overshoot.
      const int hard_cap = std::max(
          0, s.mpeg.bits_per_frame - s.sideinfo_len + s.resv_frame_start -
                 frame_used_bits - 64);
      max_bits = std::min(max_bits, hard_cap);

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
        const ShortBlockResult sbr = short_block_quantize(max_bits, ix, gr, ch, s);
        granule_bits_sum += gi.part2_3_length;
        granule_budget_sum += max_bits;
        // Fold the short granule's distortion into the frame totals so the
        // A/B block-switching compare (frame_w_dist) sees the damage a short
        // block does — above all to sustained bass through an attack —
        // instead of treating switching frames as cost-free.
        raw_signal_total += sbr.raw_signal_pow;
        raw_dist_total += sbr.raw_dist;
        w_signal_total += sbr.weighted_signal_pow;
        w_dist_total += sbr.cum_dist;
        ++granule_count;

        GranuleStats gs{};
        gs.gr = gr;
        gs.ch = ch;
        gs.gain = gi.quantizerStepSize;
        gs.compress = gi.scalefac_compress;
        gs.bits = gi.part2_3_length;
        gs.budget = max_bits;
        if (sbr.raw_signal_pow > 1.0 && sbr.raw_dist > 1.0)
          gs.raw_sdr_db =
              10.0 * std::log10(sbr.raw_signal_pow / sbr.raw_dist);
        if (sbr.weighted_signal_pow > 1.0 && sbr.cum_dist > 1.0)
          gs.perc_sdr_db =
              10.0 * std::log10(sbr.weighted_signal_pow / sbr.cum_dist);
        s.frame_granule_stats.push_back(gs);
      } else if (s.xrmax > 0) {
        // For start/stop transition blocks the spectrum is still a 576-line
        // long layout, so the long beam_search path applies; only block_type /
        // window_switching differ in the side info, set below.
        // scfsi scalefactor sharing is only valid between two normal long
        // granules; disable it whenever window switching is involved.
        // Intensity frames: ch1's sfb 11-20 scalefactor slots become per-
        // granule intensity positions, so gr0<->gr1 scalefactor sharing
        // (scfsi) must stay off for ch1.
        const int *gr0_ptr =
            (gr == 1 && gr0_sf_valid[ch] && s.block_type[ch][gr] == BLOCK_LONG &&
             !(s.intensity_frame && ch == 1))
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
          auto &lg = s.last_good_granule[ch];
          if (lg.valid) {
            gi.scalefac_compress = lg.compress;
            gi.scalefac_scale = lg.scale;
            gi.preflag = lg.preflag;
            std::copy(lg.sf.begin(), lg.sf.end(), s.scalefactor.l[gr][ch].begin());
            for (auto &v : s.side_info.scfsi[ch])
              v = 0;
          } else {
            for (auto &v : s.side_info.scfsi[ch])
              v = 0;
            gi.scalefac_compress = 0;
            gi.scalefac_scale = 0;
            gi.preflag = 0;
            for (int sfb = 0; sfb < NUM_SFB_LONG + 1; ++sfb)
              s.scalefactor.l[gr][ch][sfb] = 0;
          }
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
          // Save this successful Long/Start/Stop granule state
          auto &lg = s.last_good_granule[ch];
          lg.valid = true;
          lg.compress = gi.scalefac_compress;
          lg.scale = gi.scalefac_scale;
          lg.preflag = gi.preflag;
          std::copy(s.scalefactor.l[gr][ch].begin(), s.scalefactor.l[gr][ch].end(), lg.sf.begin());
          std::copy(s.side_info.scfsi[ch].begin(), s.side_info.scfsi[ch].end(), lg.scfsi.begin());

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
            // Preserve bass SFBs (< 200 Hz) while coarsening: reset only
            // mid/HF scalefactors so bass per-band protection survives.
            // This prevents the "boff" thump caused by bass precision
            // collapsing to zero when the forced-region recount overflows.
            int max_bass_sf = 0;
            for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
              const double lo_hz = sfbl[sfb] * s.wave.samplerate / 1152.0;
              if (lo_hz >= 200.0) {
                s.scalefactor.l[gr][ch][sfb] = 0;
              } else {
                max_bass_sf = std::max(max_bass_sf,
                                       s.scalefactor.l[gr][ch][sfb]);
              }
            }
            // Pick a compress whose slen1 can represent max_bass_sf.
            int compress = 0;
            bool found_c = false;
            for (int c = 0; c < 16; ++c) {
              const int sl = tables::slen1_tab[static_cast<std::size_t>(c)];
              const int cap = (sl > 0) ? ((1 << sl) - 1) : 0;
              if (cap >= max_bass_sf) { compress = c; found_c = true; break; }
            }
            if (!found_c && max_bass_sf > 0) {
              int max_sl = -1, best_c = 0;
              for (int c = 0; c < 16; ++c) {
                const int sl = tables::slen1_tab[static_cast<std::size_t>(c)];
                if (sl > max_sl) { max_sl = sl; best_c = c; }
              }
              compress = best_c;
            }
            gi.scalefac_compress = compress;
            gi.scalefac_scale = 0;
            gi.preflag = 0;
            gi.part2_length = 0;
            // Recompute part2 length for the surviving bass SFs.
            {
              const int sl1 = tables::slen1_tab[static_cast<std::size_t>(compress)];
              const int sl2 = tables::slen2_tab[static_cast<std::size_t>(compress)];
              // scfsi not shared for transition blocks (gr0_sf_valid is
              // cleared), so full part2 always applies.
              gi.part2_length = 6 * sl1 + 5 * sl1 + 5 * sl2 + 5 * sl2;
            }
            int step = gi.quantizerStepSize;
            for (;;) {
              const int mv = quantize(ix, step, s, gr, ch);
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
      if (s.xrmax > 0) {
        bass_dist_total += calculate_bass_dist(gr, ch, ix, s);
      }
      frame_pool_bits += std::max(0, max_bits - gi.part2_3_length);
      frame_used_bits += gi.part2_3_length;
      reservoir::adjust(gi, s);
      // +218, not the nominal +210: compensates the 12 dB front-end headroom
      // (subband promotion << 14 instead of << 16) with 2^(8/4) = 4x at the
      // decoder. Clamped to the 8-bit field; steps that high only occur on
      // the coarsen-to-zero fallback where the granule is silent anyway.
      gi.global_gain = std::min(gi.quantizerStepSize + 218, 255);
    }
  }

  // ----- Intensity-stereo finalize (Phase 1) -----
  // Runs after ch1 is fully quantized, so the decoder-implicit intensity
  // region (every sfb wholly above ch1's rzero) is known exactly. For each
  // such band write an intensity position into ch1's scalefactor slot:
  //   - sfb >= 20 (13.1 kHz+) with coherent stereo: the quantized pan angle
  //   - everything else: 7 = "intensity off" -> the band decodes as MS with
  //     S = 0 (mono), identical to the non-IS behaviour.
  // Positions 0..7 need slen2 >= 3, so ch1's scalefac_compress is upgraded
  // (9 -> 13, 14 -> 15) when needed; if the frame + reservoir cannot absorb
  // the few extra part2 bits, the whole frame stays plain MS (mode_ext = 2)
  // and nothing else changes.
  if (s.intensity_frame) {
    const auto &bt = tables::scale_fact_band_index[s.mpeg.samplerate_index];
    int boundary[MAX_GRANULES] = {NUM_SFB_LONG, NUM_SFB_LONG};
    int new_compress[MAX_GRANULES] = {0, 0};
    int deltas[MAX_GRANULES] = {0, 0};
    int delta_total = 0;
    bool ok = true;
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
      auto &gi = s.side_info.gr[gr].ch[1].tt;
      const int rz =
          std::min(gi.big_values * 2 + gi.count1 * 4, GRANULE_SAMPLES);
      int b = NUM_SFB_LONG;
      while (b > 0 && bt[b - 1] >= rz)
        --b;
      boundary[gr] = b;
      new_compress[gr] = gi.scalefac_compress;
      if (b >= NUM_SFB_LONG)
        continue;  // ch1 coded into sfb 20: no implicit region below sfb 21
      if (b < 11) {
        // Positions would spill into slen1 slots; out of Phase 1 scope.
        ok = false;
        break;
      }
      const int s2 = tables::slen2_tab[static_cast<std::size_t>(gi.scalefac_compress)];
      if (s2 < 3) {
        const int s1 = tables::slen1_tab[static_cast<std::size_t>(gi.scalefac_compress)];
        const int c_new = (s1 <= 3) ? 13 : 15;
        const int p_old = 6 * s1 + 5 * s1 + 5 * s2 + 5 * s2;
        const int s1n = tables::slen1_tab[static_cast<std::size_t>(c_new)];
        const int s2n = tables::slen2_tab[static_cast<std::size_t>(c_new)];
        const int p_new = 6 * s1n + 5 * s1n + 5 * s2n + 5 * s2n;
        new_compress[gr] = c_new;
        deltas[gr] = p_new - p_old;
        delta_total += deltas[gr];
      }
    }
    // Positions 3 (center) and 7 (off) both decode to L = R, i.e. exactly
    // what plain MS with a zeroed side already produces — a frame whose
    // every band lands there gains nothing from IS and would only pay the
    // compress-upgrade bits. Commit only when some band carries a real pan.
    int is_pos_val[MAX_GRANULES][NUM_SFB_LONG];
    bool any_pan = false;
    if (ok) {
      for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
        for (int sfb = boundary[gr]; sfb < NUM_SFB_LONG; ++sfb) {
          int pos = 7;  // intensity off -> mono, the pre-IS behaviour
          if (sfb >= 20) {
            const double aL = is_aL[gr][sfb];
            const double aR = is_aR[gr][sfb];
            if (aL + aR > 1.0 && is_co[gr][sfb] > 0.4) {
              pos = static_cast<int>(
                  std::lround(std::atan2(aL, aR) * (12.0 / 3.141592653589793)));
              pos = std::clamp(pos, 0, 6);
              if (pos != 3)
                any_pan = true;
            }
          }
          is_pos_val[gr][sfb] = pos;
        }
      }
    }
    const int cap_left = s.mpeg.bits_per_frame - s.sideinfo_len +
                         s.resv_frame_start - frame_used_bits;
    if (ok && any_pan && delta_total <= cap_left - 32) {
      for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
        if (boundary[gr] >= NUM_SFB_LONG)
          continue;
        auto &gi = s.side_info.gr[gr].ch[1].tt;
        gi.scalefac_compress = new_compress[gr];
        gi.part2_length += deltas[gr];
        gi.part2_3_length += deltas[gr];
        for (int sfb = boundary[gr]; sfb < NUM_SFB_LONG; ++sfb)
          s.scalefactor.l[gr][1][sfb] = is_pos_val[gr][sfb];
      }
      s.mpeg.mode_ext = 3;
      frame_used_bits += delta_total;
      granule_bits_sum += delta_total;
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
  s.frame_bass_dist = bass_dist_total;

  reservoir::frame_end(s);
}

} // namespace rmps::quant
