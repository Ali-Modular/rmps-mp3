#include "quant/beam_search.hpp"
#include "quant/quantize.hpp"
#include "quant/hf_rdo.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include "core/huffman_tables.hpp"
#include "core/tables.hpp"
#include "post/perceptual.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace rmps::quant {

namespace {

constexpr int    BEAM_WIDTH        = 8;
constexpr int    GAIN_WINDOW       = 1;
constexpr int    SF_MAX_CANDIDATE  = 3;
constexpr int    POLISH_ITERS      = 3;
constexpr double NOISE_SHAPING_A   = 0.5;

constexpr std::array<int, 2> COMPRESS_CANDIDATES = {9, 14};
constexpr std::array<int, 2> SCALE_CANDIDATES    = {0, 1};
constexpr std::array<int, 2> PREFLAG_CANDIDATES  = {0, 1};

constexpr std::array<std::array<int, 4>, 4> SCFSI_PATTERNS = {{
    {{0, 0, 0, 0}}, {{1, 1, 0, 0}}, {{0, 0, 1, 1}}, {{1, 1, 1, 1}},
}};

constexpr std::array<int, 5> SCFSI_BOUNDARIES = {0, 6, 11, 16, 21};
constexpr std::array<int, 22> PRETAB =
    {0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,3,3,3,2,0};

int scfsi_band_of_sfb(int sfb) {
    for (int sb = 0; sb < 4; ++sb) {
        if (SCFSI_BOUNDARIES[sb] <= sfb && sfb < SCFSI_BOUNDARIES[sb + 1])
            return sb;
    }
    return 3;
}

int part2_length_for(int compress, const std::array<int, 4>& scfsi_pat, int gr) {
    const int slen1 = tables::slen1_tab[static_cast<std::size_t>(compress)];
    const int slen2 = tables::slen2_tab[static_cast<std::size_t>(compress)];
    int b = 0;
    if (gr == 0 || !scfsi_pat[0]) b += 6 * slen1;
    if (gr == 0 || !scfsi_pat[1]) b += 5 * slen1;
    if (gr == 0 || !scfsi_pat[2]) b += 5 * slen2;
    if (gr == 0 || !scfsi_pat[3]) b += 5 * slen2;
    return b;
}

// ---- Per-band shaping quantizer (1st-order error feedback) ----
struct QuantOut {
    bool ok = false;
    std::vector<int> ix;
    int    max_val = 0;
    std::int64_t scalei_band = 0;
    double dequant_factor    = 0.0;
};

QuantOut quantize_band(const std::int32_t* xrabs_band, int n,
                       std::int64_t scalei_global, double steptab_global,
                       int sf, int scale, int preflag, int sfb,
                       const std::array<int, 10000>& int2idx,
                       double alpha) {
    QuantOut out;
    const int eff_sf = sf + (preflag ? PRETAB[static_cast<std::size_t>(sfb)] : 0);
    const int shift  = eff_sf * (1 + scale);
    const std::int64_t scalei = scalei_global << shift;
    if (scalei <= 0 || scalei > 0x7fffffffLL) return out;

    const double steptab_band   = steptab_global * static_cast<double>(1 << shift);
    const double dequant_factor = 4294967296.0 / static_cast<double>(scalei);

    out.ix.resize(static_cast<std::size_t>(n));
    int max_val = 0;
    double err = 0.0;

    for (int j = 0; j < n; ++j) {
        const std::int32_t x = xrabs_band[j];
        double target_f = (alpha > 0.0) ? (static_cast<double>(x) - alpha * err)
                                        : static_cast<double>(x);
        if (target_f < 0.0) target_f = 0.0;
        const std::int32_t target_i = static_cast<std::int32_t>(target_f);

        const std::int64_t prod = static_cast<std::int64_t>(target_i) * scalei
                                + static_cast<std::int64_t>(0x80000000LL);
        std::int32_t ln = fx::i32(prod >> 32);

        int v;
        if (ln >= 0 && ln < 10000) {
            v = int2idx[static_cast<std::size_t>(ln)];
        } else {
            const double dbl = static_cast<double>(target_i) * steptab_band * 4.656612875e-10;
            v = static_cast<int>(std::pow(std::max(dbl, 0.0), 0.75));
        }
        if (v > 8191) { out.ok = false; out.ix.clear(); return out; }
        out.ix[static_cast<std::size_t>(j)] = v;
        if (v > max_val) max_val = v;

        if (alpha > 0.0) {
            const double x_hat = (v > 0) ? std::pow(static_cast<double>(v), 4.0 / 3.0) * dequant_factor : 0.0;
            err = x_hat - target_f;
        }
    }
    out.ok = true;
    out.max_val = max_val;
    out.scalei_band = scalei;
    out.dequant_factor = dequant_factor;
    return out;
}

double band_distortion(const std::int32_t* xr_band, const std::vector<int>& ix,
                       double dequant_factor) {
    double d = 0.0;
    for (std::size_t j = 0; j < ix.size(); ++j) {
        const int v = ix[j];
        double x_hat = (v > 0) ? std::pow(static_cast<double>(v), 4.0 / 3.0) * dequant_factor : 0.0;
        const std::int32_t xr = xr_band[j];
        if (xr < 0) x_hat = -x_hat;
        const double diff = static_cast<double>(xr) - x_hat;
        d += diff * diff;
    }
    return d;
}

int bit_lower_bound(const std::vector<int>& ix) {
    int lb = 0;
    const std::size_t n = ix.size();
    for (std::size_t i = 0; i < n; i += 2) {
        const int x = ix[i];
        const int y = (i + 1 < n) ? ix[i + 1] : 0;
        if (x != 0 || y != 0) ++lb;
        if (x != 0) ++lb;
        if (y != 0) ++lb;
        if (x > 14) ++lb;
        if (y > 14) ++lb;
    }
    return lb;
}

// Lightweight stand-in passed to bit-count helpers (mirrors loop.py's GrInfoSnapshot).
struct TempGr {
    int big_values = 0;
    int count1 = 0;
    int count1table_select = 0;
    int address1 = 0;
    int address2 = 0;
    int address3 = 0;
    int region0_count = 0;
    int region1_count = 0;
    std::array<int, 3> table_select {0, 0, 0};
};

// We need to call the existing quantize helpers (calc_runlen, count1_bitcount,
// subdivide, bigv_tab_select, bigv_bitcount) which take a `GrInfo&`. Reuse
// them with a real GrInfo we throw away.
int final_bits(int* ix_full, GrInfo& tmp, EncoderState& s) {
    calc_runlen(ix_full, tmp);
    int bits = count1_bitcount(ix_full, tmp);
    subdivide(tmp, s);
    bigv_tab_select(ix_full, tmp);
    bits += bigv_bitcount(ix_full, tmp);
    return bits;
}

// Cache key (sfb, sf) -> (ix, dist, bit_lb)
struct CacheEntry { bool ok = false; std::vector<int> ix; double dist = 0.0; int bit_lb = 0; };

struct BeamState {
    std::vector<std::int8_t> sf;   // tuple of sf per stage
    double cum_dist = 0.0;
    int    cum_bit_lb = 0;
};

void materialize_ix(const std::vector<std::int8_t>& sf_path,
                    const std::vector<std::vector<CacheEntry>>& cache,
                    const std::array<int, tables::NUM_SCFB_LONG>& sfb_table,
                    int* ix_out) {
    std::fill(ix_out, ix_out + GRANULE_SAMPLES, 0);
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const int sf = sf_path[static_cast<std::size_t>(sfb)];
        const auto& e = cache[static_cast<std::size_t>(sfb)][static_cast<std::size_t>(sf)];
        const int start = sfb_table[sfb];
        for (std::size_t j = 0; j < e.ix.size(); ++j) {
            ix_out[start + static_cast<int>(j)] = e.ix[j];
        }
    }
}

} // namespace

BeamResult beam_search(int max_bits, int* ix, int gr, int ch, EncoderState& s,
                       const int* gr0_sf) {
    BeamResult out {};
    auto& cod = s.side_info.gr[gr].ch[ch].tt;
    const auto& band_table = tables::scale_fact_band_index[s.mpeg.samplerate_index];

    const bool is_side = (s.wave.channels == 2 && ch == 1
                          && (s.mpeg.mode_ext & 2));
    const auto& weights = post::perceptual_weights(
        s.mpeg.samplerate_index, s.wave.samplerate, is_side);

    // Pre-extract band slices.
    std::array<const std::int32_t*, NUM_SFB_LONG> xr_bands {};
    std::array<const std::int32_t*, NUM_SFB_LONG> xrabs_bands {};
    std::array<int, NUM_SFB_LONG> band_lens {};
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const int start = band_table[sfb];
        xr_bands[sfb]    = s.xr     + start;
        xrabs_bands[sfb] = s.xrabs.data() + start;
        band_lens[sfb]   = band_table[sfb + 1] - start;
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
    out.raw_signal_pow      = raw_signal_pow;
    out.weighted_signal_pow = weighted_signal_pow;

    // Seed gain via legacy bin-search.
    cod.scalefac_compress = COMPRESS_CANDIDATES[0];
    int seed_gain = bin_search_StepSize(max_bits, ix, cod, s);

    // R-D pre-screen window.
    std::vector<int> viable_gains;
    {
        const int budget = max_bits - part2_length_for(COMPRESS_CANDIDATES[0], {0,0,0,0}, gr);
        const int budget_max = (budget < 0) ? max_bits : budget;
        std::array<int, GRANULE_SAMPLES> tmp_ix {};
        for (int dg = -GAIN_WINDOW; dg <= GAIN_WINDOW; ++dg) {
            const int g = seed_gain + dg;
            if (g < -120 || g > 119) continue;
            if (quantize(tmp_ix.data(), g, s) > 8192) continue;
            GrInfo tmp {};
            const int bits = final_bits(tmp_ix.data(), tmp, s);
            if (bits >= 0.4 * budget_max && bits <= 1.6 * budget_max)
                viable_gains.push_back(g);
        }
        if (std::find(viable_gains.begin(), viable_gains.end(), seed_gain) == viable_gains.end()
            && seed_gain >= -120 && seed_gain <= 119)
            viable_gains.push_back(seed_gain);
        if (viable_gains.empty()) viable_gains.push_back(seed_gain);
        std::sort(viable_gains.begin(), viable_gains.end());
        viable_gains.erase(std::unique(viable_gains.begin(), viable_gains.end()),
                           viable_gains.end());
    }

    const bool mpeg1 = (s.mpeg.version == 3);
    const std::size_t scfsi_count =
        (gr == 1 && gr0_sf != nullptr && mpeg1) ? SCFSI_PATTERNS.size() : 1u;

    // Best-so-far global tracker.
    struct Best {
        std::vector<std::int8_t> sf;
        std::array<int, GRANULE_SAMPLES> ix {};
        double cum_dist = 0.0;
        std::vector<double> dist_band;
        GrInfo tmp_gi {};
        int    bits = 0;
        int    part2_length = 0;
        int    compress = 0;
        int    scale    = 0;
        int    preflag  = 0;
        std::array<int, 4> scfsi_pat {0, 0, 0, 0};
        int    gain     = 0;
    } best {};
    bool found = false;

    for (int compress : COMPRESS_CANDIDATES) {
        const int slen1 = tables::slen1_tab[static_cast<std::size_t>(compress)];
        const int slen2 = tables::slen2_tab[static_cast<std::size_t>(compress)];

        std::array<int, NUM_SFB_LONG> sf_caps {};
        for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
            const int slen = (sfb < 11) ? slen1 : slen2;
            int cap = (slen > 0) ? ((1 << slen) - 1) : 0;
            cap = std::min(cap, SF_MAX_CANDIDATE);
            sf_caps[sfb] = cap;
        }

        for (int scale : SCALE_CANDIDATES) {
            for (int preflag : PREFLAG_CANDIDATES) {
                for (int gain : viable_gains) {
                    const std::int64_t scalei_global = s.steptabi[gain + 127];
                    const double       steptab_global= s.steptab[gain + 127];

                    // Build (sfb, sf) cache once per (compress, scale, preflag, gain).
                    std::vector<std::vector<CacheEntry>> cache(NUM_SFB_LONG);
                    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
                        const int cap = sf_caps[sfb];
                        cache[static_cast<std::size_t>(sfb)].resize(static_cast<std::size_t>(cap + 1));
                        for (int sf = 0; sf <= cap; ++sf) {
                            const auto q = quantize_band(
                                xrabs_bands[sfb], band_lens[sfb],
                                scalei_global, steptab_global,
                                sf, scale, preflag, sfb, s.int2idx,
                                NOISE_SHAPING_A);
                            if (!q.ok) continue;
                            CacheEntry e;
                            e.ok = true;
                            e.ix = q.ix;
                            e.dist = band_distortion(xr_bands[sfb], q.ix, q.dequant_factor)
                                   * weights[sfb];
                            e.bit_lb = bit_lower_bound(q.ix);
                            cache[static_cast<std::size_t>(sfb)][static_cast<std::size_t>(sf)] = std::move(e);
                        }
                    }

                    for (std::size_t pi = 0; pi < scfsi_count; ++pi) {
                        const auto& scfsi_pat =
                            (gr == 1 && gr0_sf != nullptr && mpeg1)
                            ? SCFSI_PATTERNS[pi] : SCFSI_PATTERNS[0];

                        const int part2 = part2_length_for(compress, scfsi_pat, gr);
                        const int huff_budget = max_bits - part2;
                        if (huff_budget < 0) continue;

                        std::vector<BeamState> beam(1);
                        beam[0].sf.reserve(NUM_SFB_LONG);
                        bool feasible = true;

                        for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
                            const int sb = scfsi_band_of_sfb(sfb);
                            int forced_sf = -1;
                            if (gr == 1 && gr0_sf != nullptr && scfsi_pat[sb] == 1)
                                forced_sf = gr0_sf[sfb];

                            const int cap = sf_caps[sfb];
                            std::vector<BeamState> next_beam;
                            next_beam.reserve(beam.size() * (forced_sf >= 0 ? 1u : (cap + 1)));

                            for (auto& st : beam) {
                                auto try_sf = [&](int sf_try) {
                                    if (sf_try < 0) return;
                                    if (sf_try > cap && forced_sf < 0) return;
                                    auto& entry =
                                        cache[static_cast<std::size_t>(sfb)]
                                             [static_cast<std::size_t>(sf_try)];
                                    if (!entry.ok) return;
                                    const int new_lb = st.cum_bit_lb + entry.bit_lb;
                                    if (new_lb > huff_budget) return;
                                    BeamState ns;
                                    ns.sf = st.sf;
                                    ns.sf.push_back(static_cast<std::int8_t>(sf_try));
                                    ns.cum_dist = st.cum_dist + entry.dist;
                                    ns.cum_bit_lb = new_lb;
                                    next_beam.push_back(std::move(ns));
                                };
                                if (forced_sf >= 0) {
                                    if (forced_sf >= 0
                                        && forced_sf < static_cast<int>(cache[static_cast<std::size_t>(sfb)].size())
                                        && cache[static_cast<std::size_t>(sfb)][static_cast<std::size_t>(forced_sf)].ok)
                                        try_sf(forced_sf);
                                } else {
                                    for (int sf_try = 0; sf_try <= cap; ++sf_try) try_sf(sf_try);
                                }
                            }
                            if (next_beam.empty()) { feasible = false; break; }

                            std::partial_sort(next_beam.begin(),
                                              next_beam.begin()
                                                + std::min<std::size_t>(BEAM_WIDTH, next_beam.size()),
                                              next_beam.end(),
                                              [](const BeamState& a, const BeamState& b) {
                                                  return a.cum_dist < b.cum_dist;
                                              });
                            if (next_beam.size() > BEAM_WIDTH) next_beam.resize(BEAM_WIDTH);
                            beam = std::move(next_beam);
                        }
                        if (!feasible) continue;

                        // Materialise survivors and run real bit count.
                        std::array<int, GRANULE_SAMPLES> tmp_ix {};
                        for (auto& st : beam) {
                            if (found && st.cum_dist >= best.cum_dist) continue;
                            materialize_ix(st.sf, cache, band_table, tmp_ix.data());
                            GrInfo tmp_gi {};
                            const int bits = final_bits(tmp_ix.data(), tmp_gi, s);
                            if (bits > huff_budget) continue;
                            if (!found || st.cum_dist < best.cum_dist) {
                                best.sf = st.sf;
                                best.ix = tmp_ix;
                                best.cum_dist = st.cum_dist;
                                best.dist_band.resize(NUM_SFB_LONG);
                                for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
                                    best.dist_band[static_cast<std::size_t>(sfb)] =
                                        cache[static_cast<std::size_t>(sfb)]
                                             [static_cast<std::size_t>(st.sf[static_cast<std::size_t>(sfb)])].dist;
                                }
                                best.tmp_gi = tmp_gi;
                                best.bits = bits;
                                best.part2_length = part2;
                                best.compress = compress;
                                best.scale    = scale;
                                best.preflag  = preflag;
                                best.scfsi_pat = scfsi_pat;
                                best.gain      = gain;
                                found = true;
                            }
                        }
                    }
                }
            }
        }
    }

    if (!found) return out;

    // --- Local greedy ±1 polish (operates on materialised ix) ---
    {
        auto scalei_global = s.steptabi[best.gain + 127];
        auto steptab_global = s.steptab[best.gain + 127];
        const int huff_budget = max_bits - best.part2_length;
        std::array<bool, NUM_SFB_LONG> locked {};
        if (gr == 1 && gr0_sf != nullptr) {
            for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb)
                locked[sfb] = (best.scfsi_pat[scfsi_band_of_sfb(sfb)] == 1);
        }

        for (int it = 0; it < POLISH_ITERS; ++it) {
            bool improved = false;
            for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
                if (locked[sfb]) continue;
                const int slen = (sfb < 11)
                    ? tables::slen1_tab[static_cast<std::size_t>(best.compress)]
                    : tables::slen2_tab[static_cast<std::size_t>(best.compress)];
                int cap = (slen > 0) ? ((1 << slen) - 1) : 0;
                cap = std::min(cap, SF_MAX_CANDIDATE);
                const int cur_sf = best.sf[static_cast<std::size_t>(sfb)];
                for (int dlt : {-1, +1}) {
                    const int new_sf = cur_sf + dlt;
                    if (new_sf < 0 || new_sf > cap) continue;
                    auto q = quantize_band(
                        xrabs_bands[sfb], band_lens[sfb],
                        scalei_global, steptab_global,
                        new_sf, best.scale, best.preflag, sfb, s.int2idx,
                        NOISE_SHAPING_A);
                    if (!q.ok) continue;
                    const double new_dist =
                        band_distortion(xr_bands[sfb], q.ix, q.dequant_factor)
                        * post::perceptual_weights(s.mpeg.samplerate_index,
                                                   s.wave.samplerate, is_side)[sfb];
                    if (new_dist >= best.dist_band[static_cast<std::size_t>(sfb)]) continue;

                    // Apply tentatively.
                    const int start = band_table[sfb];
                    std::vector<int> saved(static_cast<std::size_t>(band_lens[sfb]));
                    for (int j = 0; j < band_lens[sfb]; ++j) {
                        saved[static_cast<std::size_t>(j)] = best.ix[start + j];
                        best.ix[start + j] = q.ix[static_cast<std::size_t>(j)];
                    }
                    GrInfo tmp_gi {};
                    const int trial_bits = final_bits(best.ix.data(), tmp_gi, s);
                    if (trial_bits <= huff_budget) {
                        best.cum_dist += (new_dist - best.dist_band[static_cast<std::size_t>(sfb)]);
                        best.dist_band[static_cast<std::size_t>(sfb)] = new_dist;
                        best.sf[static_cast<std::size_t>(sfb)] = static_cast<std::int8_t>(new_sf);
                        best.tmp_gi = tmp_gi;
                        best.bits   = trial_bits;
                        improved    = true;
                    } else {
                        for (int j = 0; j < band_lens[sfb]; ++j)
                            best.ix[start + j] = saved[static_cast<std::size_t>(j)];
                    }
                }
            }
            if (!improved) break;
        }
    }

    // --- HF rate-distortion polish ---
    const int hf_saved = hf_rdo(best.ix.data(), best.bits,
                                best.dist_band, max_bits - best.part2_length,
                                best.gain, best.scale, best.preflag,
                                xr_bands, xrabs_bands, band_lens,
                                band_table, weights, s, best.tmp_gi,
                                /*hf_start_sfb=*/11);
    out.hf_rdo_bits_saved = hf_saved;
    if (hf_saved) {
        // best.bits / cum_dist already updated by hf_rdo via reference args.
        best.cum_dist = 0.0;
        for (double d : best.dist_band) best.cum_dist += d;
    }

    // --- Compute raw distortion for stats ---
    double raw_dist = 0.0;
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const double w = weights[sfb];
        if (w > 1e-12) raw_dist += best.dist_band[static_cast<std::size_t>(sfb)] / w;
    }
    out.raw_dist = raw_dist;
    out.cum_dist = best.cum_dist;

    // --- Apply best to cod_info / scalefactor / ix output ---
    cod.scalefac_compress  = best.compress;
    cod.scalefac_scale     = best.scale;
    cod.preflag            = best.preflag;
    cod.quantizerStepSize  = best.gain;
    cod.part2_length       = best.part2_length;
    if (gr == 1 && gr0_sf != nullptr && mpeg1) {
        for (int b = 0; b < 4; ++b) s.side_info.scfsi[ch][b] = best.scfsi_pat[b];
    }
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb)
        s.scalefactor.l[gr][ch][sfb] = best.sf[static_cast<std::size_t>(sfb)];
    for (int i = 0; i < GRANULE_SAMPLES; ++i) ix[i] = best.ix[i];

    cod.big_values         = best.tmp_gi.big_values;
    cod.count1             = best.tmp_gi.count1;
    cod.count1table_select = best.tmp_gi.count1table_select;
    cod.address1           = best.tmp_gi.address1;
    cod.address2           = best.tmp_gi.address2;
    cod.address3           = best.tmp_gi.address3;
    cod.region0_count      = best.tmp_gi.region0_count;
    cod.region1_count      = best.tmp_gi.region1_count;
    cod.table_select       = best.tmp_gi.table_select;
    cod.part2_3_length     = best.part2_length + best.bits;

    out.ok             = true;
    out.part2_3_length = cod.part2_3_length;
    return out;
}

} // namespace rmps::quant
