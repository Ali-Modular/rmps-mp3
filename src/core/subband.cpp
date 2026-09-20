#include "core/subband.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include "core/scale.hpp"
#include "core/tables.hpp"

#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>

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
            // Init-time table, as in mdct.cpp: keep the historical cast.
            s.subband_fl[i][j] = fx::i32(static_cast<std::int64_t>(scaled));
        }
    }
}

// --- headroom probe (RMPS_FX_PROBE) -----------------------------------
// Counts how often each fixed-point conversion in the analysis chain actually
// saturates, and how far past the limit it reaches. The scaling convention is
// what decides this: the input enters at int16 << 16, i.e. already at ~2^31,
// and every stage below is a coherent sum.
bool g_fx_probe = (std::getenv("RMPS_FX_PROBE") != nullptr);
long long g_n_win = 0, g_clip_win = 0, g_n_fb = 0, g_clip_fb = 0;
long long g_n_in = 0, g_clip_in = 0;
std::int64_t g_max_win = 0, g_max_fb = 0, g_max_in = 0;

void window_filter(int ch, int* out, EncoderState& s, int stride) {
    std::int32_t y[64];

    auto& x_ch = s.subband_x[ch];
    int   off  = s.subband_off[ch];

    const std::int16_t* ptr = s.buffer[ch];
    std::size_t buf_idx     = s.buffer_idx[ch];

    // Replace 32 oldest samples with 32 new (in reverse). Apply input
    // pre-gain here as a multiply on the int16 sample before promotion
    // to int32 — keeps the int domain headroom intact.
    const double pre_gain = s.pre_gain;
    for (int i = 31; i >= 0; --i) {
        const double scaled = static_cast<double>(ptr[buf_idx]) * pre_gain;
        {
            // Headroom shift. int16 << 16 puts a full-scale sample at 2^31,
            // which leaves the *transform* no room: the MDCT is a coherent
            // sum, so a loud tonal bin comes out larger than the sample that
            // produced it and lands on INT32_MAX. Saturating there clips the
            // coefficient before quantization ever sees it — measured on real
            // material, bin 1 (the kick fundamental) pinned at exactly
            // 2147483647 — and no internal metric can object, because they all
            // measure the encoder's output against the already-clipped xr.
            //
            // An earlier fix in this area replaced a *wrapping* store with a
            // saturating one, which removed the sign inversion but left the
            // clip.
            //
            // Giving the transform headroom here does NOT work, and the way it
            // fails is informative: 15 (6 dB of room) measured 13.96 -> 6.03 dB
            // SNR, 14 -> 2.50. A pure input-scale change should be absorbed by
            // global_gain and cost nothing, so the encoder is *not*
            // scale-invariant — the absolute thresholds calibrated against this
            // convention (the psy model's SPL offset, XRMAX_LSB_FLOOR, the
            // silence gates) dominate any headroom gained. Fixing the clip
            // means recalibrating those, not shifting the input.
            const int hr = scale::headroom_shift();
            const std::int64_t v = static_cast<std::int64_t>(scaled) << hr;
            if (g_fx_probe) {
                ++g_n_in;
                if (v > g_max_in) g_max_in = v;
                if (-v > g_max_in) g_max_in = -v;
                if (v > 2147483647LL || v < -2147483648LL) ++g_clip_in;
            }
            x_ch[i + off] = fx::sat32(v);
        }
        buf_idx += static_cast<std::size_t>(stride);
    }
    s.buffer_idx[ch] = buf_idx;

    // Apply analysis window.
    // Accumulate in full precision and convert once. Running these sums in an
    // int32 accumulator that saturates on *every term* silently clipped any
    // partial sum that crossed the limit, and saturation is not recoverable:
    // later terms of the opposite sign bring the running value back, but the
    // clipped magnitude never returns. Both loops here are coherent sums over
    // a windowed filter, so partial sums routinely exceed the final value.
    //
    // This sits upstream of the MDCT, so it damaged everything downstream. It
    // is why a pure 1 kHz tone came back at 20.9 dB SNR at 320 kbps where
    // LAME gives 79.1, and why the error scaled with input amplitude
    // (0.005 amplitude measured 34 dB, 0.5 measured 21) instead of with the
    // bitrate — the signature of clipping, not of quantization.
    for (int i = 0; i < 64; ++i) {
        std::int64_t acc = 0;
        for (int block = 0; block < 8; ++block) {
            const int idx = (off + i + (block << 6)) & 511;
            const int win = i + (block << 6);
            acc += static_cast<std::int64_t>(x_ch[idx]) *
                   static_cast<std::int64_t>(tables::enwindow[win]);
        }
        {
            const std::int64_t v = acc >> 32;
            if (g_fx_probe) {
                ++g_n_win;
                if (v > g_max_win) g_max_win = v;
                if (-v > g_max_win) g_max_win = -v;
                if (v > 2147483647LL || v < -2147483648LL) ++g_clip_win;
            }
            y[i] = fx::sat32(v);
        }
    }

    // Slide the offset (mod HAN_SIZE).
    s.subband_off[ch] = (off + 480) & 511;

    // Apply analysis filterbank.
    for (int i = 0; i < SUBBANDS; ++i) {
        std::int64_t acc = 0;
        const auto& fl = s.subband_fl[i];
        for (int j = 0; j < 64; ++j) {
            acc += static_cast<std::int64_t>(fl[j]) *
                   static_cast<std::int64_t>(y[j]);
        }
        {
            const std::int64_t v = acc >> 32;
            if (g_fx_probe) {
                ++g_n_fb;
                if (v > g_max_fb) g_max_fb = v;
                if (-v > g_max_fb) g_max_fb = -v;
                if (v > 2147483647LL || v < -2147483648LL) ++g_clip_fb;
            }
            out[i] = fx::sat32(v);
        }
    }
}

void window_filter(int ch, int* out, EncoderState& s) {
    window_filter(ch, out, s, 1);
}

void fx_probe_report() {
    if (!g_fx_probe) return;
    std::fprintf(stderr,
                 "[fx] input      n=%lld clipped=%lld max=%lld (limit 2147483647)\n",
                 g_n_in, g_clip_in, (long long)g_max_in);
    std::fprintf(stderr, "[fx] window out n=%lld clipped=%lld max=%lld\n",
                 g_n_win, g_clip_win, (long long)g_max_win);
    std::fprintf(stderr, "[fx] filterbank n=%lld clipped=%lld max=%lld\n",
                 g_n_fb, g_clip_fb, (long long)g_max_fb);
}

} // namespace rmps::subband
