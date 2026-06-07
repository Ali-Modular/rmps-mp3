#include "quant/hf_rdo.hpp"
#include "quant/quantize.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace rmps::quant {

namespace {
constexpr double HF_RDO_DIST_FRAC = 0.0005;
constexpr std::array<int, 22> PRETAB =
    {0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,3,3,3,2,0};

int final_bits(int* ix_full, GrInfo& tmp, EncoderState& s) {
    calc_runlen(ix_full, tmp);
    int bits = count1_bitcount(ix_full, tmp);
    subdivide(tmp, s);
    bigv_tab_select(ix_full, tmp);
    bits += bigv_bitcount(ix_full, tmp);
    return bits;
}
}

int hf_rdo(int* ix_full, int& cur_bits,
           std::vector<double>& dist_band,
           std::vector<double>& raw_band,
           int huff_budget,
           int gain, int scale, int preflag,
           const std::array<const std::int32_t*, NUM_SFB_LONG>& xr_bands,
           const std::array<const std::int32_t*, NUM_SFB_LONG>& xrabs_bands,
           const std::array<int, NUM_SFB_LONG>& band_lens,
           const std::array<int, 23>& band_table,
           const std::array<double, 21>& weights,
           EncoderState& s,
           GrInfo& tmp_gi,
           int hf_start_sfb,
           int hf_end_sfb) {
    (void)xrabs_bands;
    const std::int64_t scalei_global = s.steptabi[gain + 127];

    // Weighted signal power (for per-swap cap).
    double weighted_signal = 0.0;
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        double sp = 0.0;
        for (int j = 0; j < band_lens[sfb]; ++j) {
            const double v = xr_bands[sfb][j];
            sp += v * v;
        }
        weighted_signal += sp * weights[sfb];
    }
    const double dist_cap = HF_RDO_DIST_FRAC * std::max(weighted_signal, 1.0);

    int total_saved = 0;

    const int sfb_end = std::min(hf_end_sfb, NUM_SFB_LONG);
    for (int sfb = hf_start_sfb; sfb < sfb_end; ++sfb) {
        // Effective shift = (sf + preflag*pretab) * (1+scale). We don't have sf
        // here — caller already produced ix using a given sf, so we re-derive
        // shift from gi by replaying the band's quantizer parameters via the
        // stored `dist_band` is impossible. Use a simpler model: scan the
        // already-quantized ix values and compute dequant_factor that maps
        // them back exactly. The relationship is dequant_factor = max |xr| /
        // |ix|^(4/3). Estimate from a non-zero coefficient:
        double dequant_factor = 0.0;
        for (int j = 0; j < band_lens[sfb]; ++j) {
            const int v = ix_full[band_table[sfb] + j];
            if (v > 0) {
                const double abs_xr = std::abs(static_cast<double>(xr_bands[sfb][j]));
                const double pw = std::pow(static_cast<double>(v), 4.0 / 3.0);
                if (pw > 0.0) { dequant_factor = abs_xr / pw; }
                break;
            }
        }
        if (dequant_factor <= 0.0) {
            // Fallback: derive from scalei_global directly assuming sf=0.
            dequant_factor = 4294967296.0 / static_cast<double>(scalei_global);
        }

        struct Cand { double dist_change; int j; int trial; int old; };
        std::vector<Cand> cands;
        cands.reserve(static_cast<std::size_t>(band_lens[sfb]));

        const int start = band_table[sfb];
        const double w  = weights[sfb];
        for (int j = 0; j < band_lens[sfb]; ++j) {
            const int cur_ix = ix_full[start + j];
            if (cur_ix == 0) continue;
            const double xr = xr_bands[sfb][j];
            double cur_xhat = std::pow(static_cast<double>(cur_ix), 4.0 / 3.0) * dequant_factor;
            if (xr < 0) cur_xhat = -cur_xhat;
            const double cur_err2 = (xr - cur_xhat) * (xr - cur_xhat);

            Cand best_for_j {1e30, -1, -1, cur_ix};
            for (int trial : {cur_ix - 1, 0}) {
                if (trial < 0 || trial == cur_ix) continue;
                if (trial == cur_ix - 1 && trial == 0) continue;
                double trial_xhat = (trial > 0)
                    ? std::pow(static_cast<double>(trial), 4.0 / 3.0) * dequant_factor
                    : 0.0;
                if (xr < 0) trial_xhat = -trial_xhat;
                const double trial_err2 = (xr - trial_xhat) * (xr - trial_xhat);
                const double dist_change = (trial_err2 - cur_err2) * w;
                if (dist_change > dist_cap) continue;
                if (dist_change < best_for_j.dist_change)
                    best_for_j = {dist_change, j, trial, cur_ix};
            }
            if (best_for_j.j >= 0) cands.push_back(best_for_j);
        }
        if (cands.empty()) continue;

        std::sort(cands.begin(), cands.end(),
                  [](const Cand& a, const Cand& b) { return a.dist_change < b.dist_change; });

        // Apply all tentatively.
        std::vector<std::pair<int, int>> applied;   // (j, old)
        applied.reserve(cands.size());
        double total_dist_change = 0.0;
        for (const auto& c : cands) {
            ix_full[start + c.j] = c.trial;
            applied.emplace_back(c.j, c.old);
            total_dist_change   += c.dist_change;
        }

        GrInfo trial_gi {};
        int trial_bits = final_bits(ix_full, trial_gi, s);
        int delta = cur_bits - trial_bits;

        auto accept = [&](double dist_inc) {
            cur_bits = trial_bits;
            dist_band[static_cast<std::size_t>(sfb)] += dist_inc;
            // dist_change is `(trial_err2 - cur_err2) * w`, so the raw band
            // noise delta is `dist_inc / w`. Update raw_band so callers can
            // still recover an honest raw SDR after the polish.
            if (w > 1e-12)
                raw_band[static_cast<std::size_t>(sfb)] += dist_inc / w;
            tmp_gi = trial_gi;
            total_saved += delta;
        };

        if (delta > 0 && trial_bits <= huff_budget) {
            accept(total_dist_change);
            continue;
        }
        // Retry with top half.
        const std::size_t keep_n = std::max<std::size_t>(1u, applied.size() / 2u);
        for (std::size_t k = keep_n; k < applied.size(); ++k) {
            ix_full[start + applied[k].first] = applied[k].second;
        }
        trial_bits = final_bits(ix_full, trial_gi, s);
        delta = cur_bits - trial_bits;
        if (delta > 0 && trial_bits <= huff_budget) {
            double kept_dist = 0.0;
            for (std::size_t k = 0; k < keep_n; ++k) kept_dist += cands[k].dist_change;
            accept(kept_dist);
        } else {
            // Revert top half.
            for (std::size_t k = 0; k < keep_n; ++k) {
                ix_full[start + applied[k].first] = applied[k].second;
            }
        }
    }
    return total_saved;
}

} // namespace rmps::quant
