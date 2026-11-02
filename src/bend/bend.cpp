#include "bend/bend.hpp"

#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include "core/side_info.hpp"
#include "core/tables.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace rmps::bend {

namespace {

constexpr double kAliasCoef[8] = {
    -0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037,
};

/// MDCT lines are Q31 — a full-scale sine's peak line sits near 2^31.
constexpr float kViewScale = 1.0f / 2147483648.0f;

std::uint32_t next_rand(State& b) noexcept {
    b.rng ^= b.rng << 13;
    b.rng ^= b.rng >> 17;
    b.rng ^= b.rng << 5;
    return b.rng;
}

} // namespace

int fold_source(int band, int fold) noexcept {
    fold = std::clamp(fold, 1, SUBBANDS);
    if (band < fold)
        return band;
    const int src = 2 * fold - 1 - band;
    return src >= 0 ? src : -1;
}

void butterflies(EncoderState& s, const std::int32_t* std_ca, const std::int32_t* std_cs,
                 const std::int32_t*& ca, const std::int32_t*& cs) {
    const double alias = s.bends.alias;
    if (alias == 1.0) {
        ca = std_ca;
        cs = std_cs;
        return;
    }
    State& b = s.bend;
    if (!b.tables_valid || b.tables_alias != alias) {
        // Each butterfly is a rotation by atan(c): ca = sin, cs = cos. The
        // bend scales the angle, so 0 is the identity and 1 the standard.
        for (int i = 0; i < 8; ++i) {
            const double a = alias * std::atan(kAliasCoef[i]);
            b.ca[i] = fx::sat32(std::sin(a) * 2147483647.0);
            b.cs[i] = fx::sat32(std::cos(a) * 2147483647.0);
        }
        b.tables_alias = alias;
        b.tables_valid = true;
    }
    ca = b.ca.data();
    cs = b.cs.data();
}

void spectrum(EncoderState& s, int ch, int gr) {
    if (s.block_type[ch][gr] == BLOCK_SHORT)
        return;   // interleaved windows: no line-for-line meaning to bend

    State& b = s.bend;
    const EncoderBends& e = s.bends;
    auto& xr = s.mdct_freq[ch][gr];

    for (int i = 0; i < GRANULE_SAMPLES; ++i)
        b.building.before[i] += std::fabs(static_cast<float>(xr[i])) * kViewScale;

    // Warp: line i takes line i / warp. Read from a copy, since the read
    // index runs ahead of the write one when warp < 1.
    if (e.warp != 1.0 && e.warp > 0.0) {
        const auto src = xr;
        const double inv = 1.0 / e.warp;
        for (int i = 0; i < GRANULE_SAMPLES; ++i) {
            const int j = static_cast<int>(i * inv);
            xr[i] = (j < GRANULE_SAMPLES) ? src[j] : 0;
        }
    }

    // Smear and freeze: a held magnitude per line, the live sign on top. A
    // line the live signal has nothing in takes a random sign, or a frozen
    // spectrum over silence would come out as one repeating impulse.
    const double smear = std::clamp(e.smear, 0.0, 0.9999);
    if (smear > 0.0 || e.freeze) {
        auto& held = b.held[ch];
        const float decay = static_cast<float>(smear);
        for (int i = 0; i < GRANULE_SAMPLES; ++i) {
            const std::int32_t v = xr[i];
            const float m = std::fabs(static_cast<float>(v));
            if (!e.freeze)
                held[i] = std::max(m, held[i] * decay);
            const bool neg = v < 0 || (v == 0 && (next_rand(b) & 1u));
            const double h = held[i];
            xr[i] = fx::sat32(neg ? -h : h);
        }
    } else {
        // Keep the hold tracking, so freezing captures the present rather
        // than whatever was held when the bends were last on.
        auto& held = b.held[ch];
        for (int i = 0; i < GRANULE_SAMPLES; ++i)
            held[i] = std::fabs(static_cast<float>(xr[i]));
    }
    ++b.view_granules;
}

void prune(EncoderState& s, int ch, int gr) {
    const double range_db = s.bends.mask_range_db;
    if (!(range_db < 200.0) || s.block_type[ch][gr] == BLOCK_SHORT)
        return;
    const auto& mt = s.mask_thresholds[ch][gr];
    const auto& sfb = tables::scale_fact_band_index[s.mpeg.samplerate_index];
    auto& xr = s.mdct_freq[ch][gr];

    // How far each band stands over its threshold, and the most any does.
    std::array<double, NUM_SFB_LONG> margin{};
    double best = 0.0;
    for (int b = 0; b < NUM_SFB_LONG; ++b) {
        margin[b] = mt.thr[b] > 0.0 ? mt.e_band[b] / mt.thr[b] : 0.0;
        best = std::max(best, margin[b]);
    }
    if (best <= 0.0)
        return;
    const double floor = best * std::pow(10.0, -std::max(range_db, 0.0) / 10.0);

    bool last_pruned = false;
    for (int b = 0; b < NUM_SFB_LONG; ++b) {
        last_pruned = margin[b] < floor;
        if (last_pruned)
            std::fill(xr.begin() + sfb[b], xr.begin() + sfb[b + 1], 0);
    }
    // sfb 21 carries no threshold of its own; it goes with the band below it.
    if (last_pruned)
        std::fill(xr.begin() + sfb[NUM_SFB_LONG], xr.end(), 0);
}

void coded(EncoderState& s, int ch, int gr, const int* ix) {
    if (s.block_type[ch][gr] == BLOCK_SHORT || ch > 1)
        return;
    State& b = s.bend;
    auto& out = b.coded[gr][ch];
    out.fill(0.f);
    b.coded_valid[gr][ch] = true;

    const auto& gi = s.side_info.gr[gr].ch[ch].tt;
    const std::int32_t scalei = s.steptabi[std::clamp(gi.quantizerStepSize + 127, 0, 255)];
    if (scalei <= 0)
        return;
    // A decoder's requantization, in the encoder's Q31 units: the line was
    // quantized as (|xr| * 2^sf * scalei / 2^32)^(3/4), so it comes back as
    // |ix|^(4/3) * 2^32 / scalei, less the scalefactor's amplification.
    const double step = 4294967296.0 / static_cast<double>(scalei);
    const double sf_mult = gi.scalefac_scale ? 1.0 : 0.5;
    const auto& sfb = tables::scale_fact_band_index[s.mpeg.samplerate_index];
    const auto& sf = s.scalefactor.l[gr][ch];
    const auto& xr = s.mdct_freq[ch][gr];
    // Past big_values and count1 nothing is sent, whatever ix still holds.
    const int end = std::min(GRANULE_SAMPLES, gi.big_values * 2 + gi.count1 * 4);
    for (int band = 0; band < NUM_SFB_LONG + 1; ++band) {
        const int amp = sf[band] + (gi.preflag && band < NUM_SFB_LONG ? tables::pretab[band] : 0);
        const double k = step * std::exp2(-sf_mult * amp) * kViewScale;
        for (int i = sfb[band]; i < std::min(sfb[band + 1], end); ++i) {
            const int q = std::min(std::abs(ix[i]), 8191);
            if (q == 0)
                continue;
            const double v = tables::pow43_tab[static_cast<std::size_t>(q)] * k;
            out[i] = static_cast<float>(xr[i] < 0 ? -v : v);
        }
    }
}

int starve(const EncoderState& s, int max_bits) noexcept {
    const double k = std::clamp(s.bends.starve, 0.0, 1.0);
    if (k >= 1.0)
        return max_bits;
    // A floor under the grant. With none at all the rate loop drives the step
    // size to its ceiling looking for a zero-bit coding, the all-zero guard
    // then restores the granule's loudest line at that step, and a starved
    // stream comes out *louder* - one full-scale line per granule. A few
    // dozen bits is enough to keep the step size where the signal is.
    constexpr int kFloorBits = 40;
    return std::max(static_cast<int>(max_bits * k), std::min(max_bits, kFloorBits));
}

void account(EncoderState& s, int granted, int used) noexcept {
    s.bend.building.bits_granted += granted;
    s.bend.building.bits_used += used;
}

int window(const EncoderState& s, int desired) noexcept {
    switch (s.bends.window) {
    case 1:  return BLOCK_LONG;
    case 2:  return BLOCK_SHORT;
    default: return desired;
    }
}

void begin_frame(EncoderState& s) noexcept {
    State& b = s.bend;
    b.building.before.fill(0.f);
    b.building.after.fill(0.f);
    b.building.bits_used = 0;
    b.building.bits_granted = 0;
    b.building.short_blocks = false;
    b.view_granules = 0;
    for (auto& g : b.coded_valid)
        g.fill(false);
}

void end_frame(EncoderState& s) noexcept {
    State& b = s.bend;

    // The coded spectra, back to L/R where the frame went out as M/S.
    const bool ms = s.wave.channels == 2 && (s.mpeg.mode_ext & 2) != 0;
    constexpr float kInvSqrt2 = 0.70710678f;
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
        auto& c = b.coded[gr];
        const auto& valid = b.coded_valid[gr];
        if (ms && valid[0] && valid[1]) {
            for (int i = 0; i < GRANULE_SAMPLES; ++i) {
                const float m = c[0][i], d = c[1][i];
                b.building.after[i] += (std::fabs(m + d) + std::fabs(m - d)) * kInvSqrt2;
            }
        } else {
            for (int ch = 0; ch < 2; ++ch)
                if (valid[ch])
                    for (int i = 0; i < GRANULE_SAMPLES; ++i)
                        b.building.after[i] += std::fabs(c[ch][i]);
        }
    }

    if (b.view_granules > 0) {
        const float k = 1.0f / static_cast<float>(b.view_granules);
        for (int i = 0; i < GRANULE_SAMPLES; ++i) {
            b.building.before[i] *= k;
            b.building.after[i] *= k;
        }
    }
    for (int ch = 0; ch < s.wave.channels; ++ch)
        for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr)
            if (s.block_type[ch][gr] == BLOCK_SHORT)
                b.building.short_blocks = true;
    b.view = b.building;
}

} // namespace rmps::bend
