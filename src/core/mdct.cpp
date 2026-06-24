#include "core/mdct.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include "core/subband.hpp"
#include "core/tables.hpp"

#include <array>
#include <cmath>

namespace rmps::mdct {

namespace {

constexpr double PI = 3.141592653589793;
constexpr double PI36 = PI / 36.0;
constexpr double PI72 = PI / 72.0;
constexpr double PI12 = PI / 12.0;
constexpr double PI6  = PI / 6.0;

inline std::int32_t to_q31(double v) {
  return fx::i32(static_cast<std::int64_t>(v * static_cast<double>(0x7fffffff)));
}

// Long-block analysis window value at index k (0..35): sin(pi/36 (k+0.5)).
inline double win_long(int k) { return std::sin(PI36 * (k + 0.5)); }

// Start (block_type 1, long->short) 36-point window.
inline double win_start(int k) {
  if (k < 18) return std::sin(PI36 * (k + 0.5));
  if (k < 24) return 1.0;
  if (k < 30) return std::sin(PI12 * (k - 18 + 0.5));
  return 0.0;
}

// Stop/end (block_type 3, short->long) 36-point window.
inline double win_stop(int k) {
  if (k < 6) return 0.0;
  if (k < 12) return std::sin(PI12 * (k - 6 + 0.5));
  if (k < 18) return 1.0;
  return std::sin(PI36 * (k + 0.5));
}

// Pure 36->18 long MDCT cosine basis (no window): cos(pi/72 (2k+19)(2m+1)).
inline double basis_l(int m, int k) {
  return std::cos(PI72 * (2.0 * k + 19.0) * (2.0 * m + 1.0));
}

// Anti-aliasing butterfly coefficients (per Encoder).
constexpr std::array<double, 8> AA_COEF = {
    -0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037,
};

inline std::int32_t make_ca(double c) {
  return fx::i32(static_cast<std::int64_t>(c / std::sqrt(1.0 + c * c) *
                                           static_cast<double>(0x7fffffff)));
}
inline std::int32_t make_cs(double c) {
  return fx::i32(static_cast<std::int64_t>(1.0 / std::sqrt(1.0 + c * c) *
                                           static_cast<double>(0x7fffffff)));
}

std::array<std::int32_t, 8> g_ca{};
std::array<std::int32_t, 8> g_cs{};
bool g_aa_inited = false;

void init_aa() {
  if (g_aa_inited)
    return;
  for (int i = 0; i < 8; ++i) {
    g_ca[i] = make_ca(AA_COEF[i]);
    g_cs[i] = make_cs(AA_COEF[i]);
  }
  g_aa_inited = true;
}

inline void cmuls(std::int32_t a_re, std::int32_t a_im, std::int32_t b_re,
                  std::int32_t b_im, std::int32_t &d_re, std::int32_t &d_im) {
  const std::int64_t tre = (static_cast<std::int64_t>(a_re) * b_re -
                            static_cast<std::int64_t>(a_im) * b_im) >>
                           31;
  const std::int64_t tim = (static_cast<std::int64_t>(a_re) * b_im +
                            static_cast<std::int64_t>(a_im) * b_re) >>
                           31;
  d_re = fx::i32(tre);
  d_im = fx::i32(tim);
}

} // namespace

void initialise(EncoderState &s) {
  init_aa();
  for (int m = 0; m < K_PER_GRANULE; ++m) {
    for (int k = 0; k < 36; ++k) {
      const double b = basis_l(m, k);
      // Long (unchanged): window folded into the basis, bit-identical to before.
      s.mdct_cos_l[m][k] = to_q31(win_long(k) * b);
      // Transition blocks: same basis, different window shape.
      s.mdct_cos_start[m][k] = to_q31(win_start(k) * b);
      s.mdct_cos_stop[m][k]  = to_q31(win_stop(k) * b);
    }
  }

  // Short MDCT basis (12 inputs -> 6 outputs) with the short sine window
  // folded in: w_s[n] = sin(pi/12 (n+0.5)); basis = cos(pi/6 (n+3.5)(k+0.5)).
  for (int k = 0; k < 6; ++k) {
    for (int n = 0; n < 12; ++n) {
      const double w_s = std::sin(PI12 * (n + 0.5));
      const double b   = std::cos(PI6 * (n + 3.5) * (k + 0.5));
      s.mdct_cos_s[k][n] = to_q31(w_s * b);
    }
  }

  // Short-block reorder table: transmission order (sfb, window, freq-line)
  // -> natural per-subband short MDCT position (sb*18 + w*6 + k). Within a
  // window, frequency line f maps to subband f/6 and intra-subband index f%6.
  {
    const auto &sb_s = tables::scale_fact_band_index_short[s.mpeg.samplerate_index];
    int ti = 0;
    for (int sfb = 0; sfb < tables::NUM_SCFB_SHORT - 1; ++sfb) {
      for (int w = 0; w < NUM_WINDOWS; ++w) {
        for (int f = sb_s[sfb]; f < sb_s[sfb + 1]; ++f) {
          const int sb = f / 6;
          const int kk = f % 6;
          s.short_reorder[ti++] = sb * 18 + w * 6 + kk;
        }
      }
    }
  }
}

void analyse(EncoderState &s, int stride) {
  for (int ch = 0; ch < s.wave.channels; ++ch) {
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
      // Polyphase filtering — two subband blocks at a time.
      for (int k = 0; k < K_PER_GRANULE; k += 2) {
        int row0[32], row1[32];
        subband::window_filter(ch, row0, s, stride);
        subband::window_filter(ch, row1, s, stride);
        for (int b = 0; b < SUBBANDS; ++b) {
          s.l3_sb_sample[ch][gr + 1][k][b] = row0[b];
          s.l3_sb_sample[ch][gr + 1][k + 1][b] = row1[b];
        }
        // Compensate filter polarity on odd subband indices (analysis side).
        for (int b = 1; b < SUBBANDS; b += 2) {
          s.l3_sb_sample[ch][gr + 1][k + 1][b] =
              -s.l3_sb_sample[ch][gr + 1][k + 1][b];
        }
      }

      const int bt = s.block_type[ch][gr];

      if (bt == BLOCK_SHORT) {
        // ----- Short blocks: three 12->6 MDCTs per subband -----
        // Produce coefficients in *natural* order (sb*18 + w*6 + k), then
        // scatter into transmission order via short_reorder. No anti-aliasing
        // butterfly is applied to short blocks.
        std::array<std::int32_t, GRANULE_SAMPLES> natural{};
        for (int band = 0; band < SUBBANDS; ++band) {
          // 36-sample input window: prev granule (0..17) + current (18..35).
          std::int32_t in36[36];
          for (int k = 0; k < K_PER_GRANULE; ++k) {
            in36[k]      = s.l3_sb_sample[ch][gr][k][band];
            in36[k + 18] = s.l3_sb_sample[ch][gr + 1][k][band];
          }
          for (int w = 0; w < NUM_WINDOWS; ++w) {
            const std::int32_t *sub = in36 + 6 + 6 * w; // 12 samples, hop 6
            for (int k = 0; k < 6; ++k) {
              // Accumulate the 12-tap short MDCT in full precision. The forward
              // MDCT's TDAC normalisation is 2/n, so the short transform (n=12)
              // must be 3x the long convention (n=36) to share the same
              // global_gain / dequant scale — otherwise short blocks decode
              // ~9 dB quieter than the surrounding long blocks. Fold that
              // factor of 3 into the final shift (acc holds 2^31 * raw_sum;
              // (acc*3)>>32 == 3 * (raw_sum/2)).
              std::int64_t acc = 0;
              const auto &cos_k = s.mdct_cos_s[k];
              for (int n = 0; n < 12; ++n) {
                acc += static_cast<std::int64_t>(sub[n]) * cos_k[n];
              }
              natural[band * 18 + w * 6 + k] = fx::i32((acc * 3) >> 32);
            }
          }
        }
        auto &out = s.mdct_freq[ch][gr];
        for (int ti = 0; ti < GRANULE_SAMPLES; ++ti)
          out[ti] = natural[s.short_reorder[ti]];
      } else {
        // ----- Long / start / stop: 36->18 MDCT with the matching window -----
        const auto &cos_tab = (bt == BLOCK_START) ? s.mdct_cos_start
                              : (bt == BLOCK_STOP) ? s.mdct_cos_stop
                                                   : s.mdct_cos_l;
        std::int32_t mdct_in[36];
        for (int band = 0; band < SUBBANDS; ++band) {
          for (int k = 0; k < K_PER_GRANULE; ++k) {
            mdct_in[k] = s.l3_sb_sample[ch][gr][k][band];
            mdct_in[k + 18] = s.l3_sb_sample[ch][gr + 1][k][band];
          }
          for (int k = 0; k < K_PER_GRANULE; ++k) {
            std::int32_t vm = 0;
            const auto &cos_k = cos_tab[k];
            for (int j = 0; j < 36; ++j) {
              vm = fx::i32(static_cast<std::int64_t>(vm) +
                           fx::mul(mdct_in[j], cos_k[j]));
            }
            s.mdct_freq[ch][gr][band * 18 + k] = vm;
          }

          // Anti-aliasing butterfly (except for band 0).
          if (band != 0) {
            for (int i = 0; i < 8; ++i) {
              const int idx_curr = band * 18 + i;
              const int idx_prev = (band - 1) * 18 + (17 - i);
              std::int32_t dre, dim;
              cmuls(s.mdct_freq[ch][gr][idx_curr], s.mdct_freq[ch][gr][idx_prev],
                    g_cs[i], g_ca[i], dre, dim);
              s.mdct_freq[ch][gr][idx_curr] = dre;
              s.mdct_freq[ch][gr][idx_prev] = dim;
            }
          }
        }
      }
    }

    // Slide latest granule's subband samples back into slot 0.
    const int src = s.mpeg.granules_per_frame;
    for (int k = 0; k < K_PER_GRANULE; ++k) {
      for (int b = 0; b < SUBBANDS; ++b) {
        s.l3_sb_sample[ch][0][k][b] = s.l3_sb_sample[ch][src][k][b];
      }
    }
  }
}

} // namespace rmps::mdct
