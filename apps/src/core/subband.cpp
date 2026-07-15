#include "core/subband.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include "core/tables.hpp"

#include <cmath>

namespace rmps::subband {

namespace {
constexpr double PI64 = 3.141592653589793 / 64.0;
}

void initialise(EncoderState& s) {
    s.subband_off = {0, 0};
    for (auto& row : s.subband_x)  row.fill(0);
    for (auto& row : s.subband_fl) row.fill(0);

    for (int i = 0; i < SUBBANDS; ++i) {
        for (int j = 0; j < 64; ++j) {
            double f = 1e9 * std::cos((2.0 * i + 1.0) * (16.0 - j) * PI64);
            f = (f >= 0.0) ? std::floor(f + 0.5) : std::ceil(f - 0.5);
            const double scaled = f * (static_cast<double>(0x7fffffff) * 1e-9);
            s.subband_fl[i][j] = fx::i32(static_cast<std::int64_t>(scaled));
        }
    }
}

void window_filter(int ch, int* out, EncoderState& s, int stride) {
    std::int32_t y[64];

    auto& x_ch = s.subband_x[ch];
    int   off  = s.subband_off[ch];

    const std::int16_t* ptr = s.buffer[ch];
    std::size_t buf_idx     = s.buffer_idx[ch];

    // Replace 32 oldest samples with 32 new (in reverse). Apply input
    // pre-gain here as a multiply on the int16 sample before promotion
    // to int32 — keeps the int domain headroom intact.
    //
    // Promotion is << 14, not << 16: the analysis filterbank + MDCT have a
    // per-bin gain above 1 for a tone concentrated in a single line (~1.2x
    // measured on fullscale bass), so a 0 dBFS master promoted to full int32
    // scale clips the MDCT's int32 clamp — decoded as loud low-frequency
    // garbage on every bass peak. The 12 dB headroom here is compensated
    // exactly at the decoder by +8 on global_gain (2^(8/4) = 4x).
    const double pre_gain = s.pre_gain;
    for (int i = 31; i >= 0; --i) {
        const double scaled = static_cast<double>(ptr[buf_idx]) * pre_gain;
        x_ch[i + off] = fx::i32(static_cast<std::int64_t>(scaled) << 14);
        buf_idx += static_cast<std::size_t>(stride);
    }
    s.buffer_idx[ch] = buf_idx;

    // Apply analysis window.
    for (int i = 0; i < 64; ++i) {
        std::int32_t sv = 0;
        for (int block = 0; block < 8; ++block) {
            const int idx = (off + i + (block << 6)) & 511;
            const int win = i + (block << 6);
            sv = fx::i32(static_cast<std::int64_t>(sv)
                       + fx::mul(x_ch[idx], tables::enwindow[win]));
        }
        y[i] = sv;
    }

    // Slide the offset (mod HAN_SIZE).
    s.subband_off[ch] = (off + 480) & 511;

    // Apply analysis filterbank.
    for (int i = 0; i < SUBBANDS; ++i) {
        std::int32_t sv = 0;
        const auto& fl = s.subband_fl[i];
        for (int j = 0; j < 64; ++j) {
            sv = fx::i32(static_cast<std::int64_t>(sv) + fx::mul(fl[j], y[j]));
        }
        out[i] = sv;
    }
}

void window_filter(int ch, int* out, EncoderState& s) {
    window_filter(ch, out, s, 1);
}

} // namespace rmps::subband // trigger rebuild
