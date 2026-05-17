#include "core/mdct.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include "core/subband.hpp"

#include <array>
#include <cmath>

namespace rmps::mdct {

namespace {

constexpr double PI = 3.141592653589793;
constexpr double PI36 = PI / 36.0;
constexpr double PI72 = PI / 72.0;

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
      const double v = std::sin(PI36 * (k + 0.5)) *
                       std::cos(PI72 * (2.0 * k + 19.0) * (2.0 * m + 1.0));
      s.mdct_cos_l[m][k] = fx::i32(
          static_cast<std::int64_t>(v * static_cast<double>(0x7fffffff)));
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

      // 18-point MDCT per subband.
      std::int32_t mdct_in[36];
      for (int band = 0; band < SUBBANDS; ++band) {
        for (int k = 0; k < K_PER_GRANULE; ++k) {
          mdct_in[k] = s.l3_sb_sample[ch][gr][k][band];
          mdct_in[k + 18] = s.l3_sb_sample[ch][gr + 1][k][band];
        }
        for (int k = 0; k < K_PER_GRANULE; ++k) {
          std::int32_t vm = 0;
          const auto &cos_k = s.mdct_cos_l[k];
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
