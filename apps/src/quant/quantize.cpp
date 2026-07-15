#include "quant/quantize.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include "core/huffman_tables.hpp"
#include "core/tables.hpp"

#include <algorithm>
#include <climits>
#include <cmath>

namespace rmps::quant {

namespace {

constexpr std::array<std::pair<int, int>, 23> SUBDV_TABLE = {{
    {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
    {0, 1}, {1, 1}, {1, 1}, {1, 2}, {2, 2},
    {2, 3}, {2, 3}, {3, 4}, {3, 4}, {3, 4},
    {4, 5}, {4, 5}, {4, 6}, {5, 6}, {5, 6},
    {5, 7}, {6, 7}, {6, 7},
}};

} // namespace

void initialise(EncoderState& s) {
    for (int i = 0; i < 256; ++i) {
        s.steptab[i] = std::pow(2.0, (127.0 - i) / 4.0);
        const double v = s.steptab[i] * 2.0;
        s.steptabi[i] = (v > 0x7fffffff)
            ? 0x7fffffff
            : (v < 0.5 ? 0 : static_cast<std::int32_t>(v + 0.5));
    }
    for (int i = 0; i < 262144; ++i) {
        s.int2idx[i] = static_cast<int>(std::pow(static_cast<double>(i), 0.75) - 0.0946 + 0.5);
    }
}

int quantize(int* ix, int stepsize, EncoderState& s, int gr, int ch) {
    const std::int32_t scalei_global = s.steptabi[std::clamp(stepsize + 127, 0, 255)];
    const auto& sfbl = tables::scale_fact_band_index[s.mpeg.samplerate_index];
    const int scale = s.side_info.gr[gr].ch[ch].tt.scalefac_scale;
    const int preflag = s.side_info.gr[gr].ch[ch].tt.preflag;

    int max_val = 0;

    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const int start = sfbl[sfb];
        const int end = sfbl[sfb + 1];

        const int sf = s.scalefactor.l[gr][ch][sfb];
        const int eff_sf = sf + (preflag ? tables::pretab[static_cast<std::size_t>(sfb)] : 0);
        const int idx = std::clamp((1 + scale) * eff_sf, 0, 39);
        const double sf_mult = tables::pow2_half_tab[static_cast<std::size_t>(idx)];

        const std::int64_t scalei = static_cast<std::int64_t>(static_cast<double>(scalei_global) * sf_mult + 0.5);

        if (scalei <= 0) {
            for (int i = start; i < end; ++i) {
                ix[i] = 0;
            }
            continue;
        }

        for (int i = start; i < end; ++i) {
            const std::int32_t x = s.xrabs[i];
            if (x == 0) {
                ix[i] = 0;
                continue;
            }
            const std::int64_t prod = static_cast<std::int64_t>(x) * scalei + 0x80000000LL;
            const std::int32_t ln = static_cast<std::int32_t>(prod >> 32);
            if (ln > 165140) return 16384;
            int v;
            if (ln < 262144) {
                v = s.int2idx[static_cast<std::size_t>(ln)];
            } else {
                v = 8191;
            }
            ix[i] = v;
            if (v > max_val) max_val = v;
        }
    }

    // sfb 21
    {
        const int start = sfbl[NUM_SFB_LONG];
        const int end = GRANULE_SAMPLES;

        const std::int64_t scalei = scalei_global;
        if (scalei <= 0) {
            for (int i = start; i < end; ++i) {
                ix[i] = 0;
            }
        } else {
            const std::int32_t scalei_32 = (scalei > 0x7fffffffLL) ? 0x7fffffff : static_cast<std::int32_t>(scalei);
            for (int i = start; i < end; ++i) {
                const std::int32_t x = s.xrabs[i];
                if (x == 0) {
                    ix[i] = 0;
                    continue;
                }
                const std::int32_t ln = fx::mulr(x, scalei_32);
                if (ln > 165140) return 16384;
                int v;
                if (ln < 262144) {
                    v = s.int2idx[static_cast<std::size_t>(ln)];
                } else {
                    v = 8191;
                }
                ix[i] = v;
                if (v > max_val) max_val = v;
            }
        }
    }

    return max_val;
}

void calc_runlen(int* ix, GrInfo& gi) {
    int i = GRANULE_SAMPLES;
    while (i > 1) {
        if (ix[i - 1] == 0 && ix[i - 2] == 0) i -= 2;
        else break;
    }
    gi.count1 = 0;
    while (i > 3) {
        if (ix[i - 1] <= 1 && ix[i - 2] <= 1 && ix[i - 3] <= 1 && ix[i - 4] <= 1) {
            gi.count1 += 1;
            i -= 4;
        } else break;
    }
    gi.big_values = i >> 1;
}

int count1_bitcount(int* ix, GrInfo& gi) {
    static constexpr std::array<int, 16> POPCOUNT_16 = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    };
    int sum0 = 0, sum1 = 0;
    const int start_idx = gi.big_values << 1;
    const auto* h32 = huffman::at(32).hlen;
    const auto* h33 = huffman::at(33).hlen;
    for (int k = 0; k < gi.count1; ++k) {
        const int i = start_idx + k * 4;
        const int v = ix[i], w = ix[i+1], x = ix[i+2], y = ix[i+3];
        const int p = v + (w << 1) + (x << 2) + (y << 3);
        const int signbits = POPCOUNT_16[p];
        sum0 += signbits + h32[p];
        sum1 += signbits + h33[p];
    }
    if (sum0 < sum1) { gi.count1table_select = 0; return sum0; }
    gi.count1table_select = 1; return sum1;
}

void subdivide(GrInfo& gi, EncoderState& s) {
    if (!gi.big_values) {
        gi.region0_count = 0;
        gi.region1_count = 0;
        return;
    }
    const auto& sfb = tables::scale_fact_band_index[s.mpeg.samplerate_index];
    const int bigvalues_region = 2 * gi.big_values;

    int scfb_anz = 0;
    while (sfb[scfb_anz] < bigvalues_region) ++scfb_anz;

    int thiscount = SUBDV_TABLE[scfb_anz].first;
    while (thiscount > 0) {
        if (sfb[thiscount + 1] <= bigvalues_region) break;
        --thiscount;
    }
    gi.region0_count = thiscount;
    gi.address1      = sfb[thiscount + 1];

    int idx_offset = gi.region0_count + 1;
    thiscount = SUBDV_TABLE[scfb_anz].second;
    while (thiscount > 0) {
        if (sfb[idx_offset + thiscount + 1] <= bigvalues_region) break;
        --thiscount;
    }
    gi.region1_count = thiscount;
    gi.address2      = sfb[idx_offset + thiscount + 1];
    gi.address3      = bigvalues_region;
}

int ix_max(int* ix, int begin, int end) {
    int m = 0;
    for (int i = begin; i < end; ++i) if (ix[i] > m) m = ix[i];
    return m;
}

int count_bit(int* ix, int start, int end, int table) {
    if (!table) return 0;
    const auto& h = huffman::at(table);
    int sum_bits = 0;
    if (table > 15) {
        for (int i = start; i < end; i += 2) {
            int x = ix[i], y = ix[i + 1];
            if (x > 14) { x = 15; sum_bits += h.linbits; }
            if (y > 14) { y = 15; sum_bits += h.linbits; }
            sum_bits += h.hlen[x * h.ylen + y];
            if (x) ++sum_bits;
            if (y) ++sum_bits;
        }
    } else {
        for (int i = start; i < end; i += 2) {
            const int x = ix[i], y = ix[i + 1];
            sum_bits += h.hlen[x * h.ylen + y];
            if (x) ++sum_bits;
            if (y) ++sum_bits;
        }
    }
    return sum_bits;
}

int new_choose_table(int* ix, int begin, int end) {
    int mv = ix_max(ix, begin, end);
    if (!mv) return 0;

    int choice[2] = {0, 0};
    int sum_bits[2] = {0, 0};

    if (mv < 15) {
        for (int i = 13; i >= 0; --i) {
            if (huffman::at(i).xlen > mv) { choice[0] = i; break; }
        }
        sum_bits[0] = count_bit(ix, begin, end, choice[0]);
        switch (choice[0]) {
            case 2: {
                int s1 = count_bit(ix, begin, end, 3);
                if (s1 <= sum_bits[0]) choice[0] = 3;
                break;
            }
            case 5: {
                int s1 = count_bit(ix, begin, end, 6);
                if (s1 <= sum_bits[0]) choice[0] = 6;
                break;
            }
            case 7: {
                int s1 = count_bit(ix, begin, end, 8);
                if (s1 <= sum_bits[0]) { choice[0] = 8; sum_bits[0] = s1; }
                int s2 = count_bit(ix, begin, end, 9);
                if (s2 <= sum_bits[0]) choice[0] = 9;
                break;
            }
            case 10: {
                int s1 = count_bit(ix, begin, end, 11);
                if (s1 <= sum_bits[0]) { choice[0] = 11; sum_bits[0] = s1; }
                int s2 = count_bit(ix, begin, end, 12);
                if (s2 <= sum_bits[0]) choice[0] = 12;
                break;
            }
            case 13: {
                int s1 = count_bit(ix, begin, end, 15);
                if (s1 <= sum_bits[0]) choice[0] = 15;
                break;
            }
            default: break;
        }
    } else {
        mv -= 15;
        for (int i = 15; i < 24; ++i) if (huffman::at(i).linmax >= mv) { choice[0] = i; break; }
        for (int i = 24; i < 32; ++i) if (huffman::at(i).linmax >= mv) { choice[1] = i; break; }
        sum_bits[0] = count_bit(ix, begin, end, choice[0]);
        sum_bits[1] = count_bit(ix, begin, end, choice[1]);
        if (sum_bits[1] < sum_bits[0]) choice[0] = choice[1];
    }
    return choice[0];
}

int bigv_bitcount(int* ix, const GrInfo& gi) {
    int bits = 0;
    if (gi.table_select[0]) bits += count_bit(ix, 0,                gi.address1, gi.table_select[0]);
    if (gi.table_select[1]) bits += count_bit(ix, gi.address1,      gi.address2, gi.table_select[1]);
    if (gi.table_select[2]) bits += count_bit(ix, gi.address2,      gi.address3, gi.table_select[2]);
    return bits;
}

void bigv_tab_select(int* ix, GrInfo& gi) {
    gi.table_select = {0, 0, 0};
    if (gi.address1 > 0)
        gi.table_select[0] = new_choose_table(ix, 0,           gi.address1);
    if (gi.address2 > gi.address1)
        gi.table_select[1] = new_choose_table(ix, gi.address1, gi.address2);
    if ((gi.big_values << 1) > gi.address2)
        gi.table_select[2] = new_choose_table(ix, gi.address2, gi.big_values << 1);
}

int best_subdivide_rd(int* ix, GrInfo& gi, EncoderState& s) {
    const int bigv = gi.big_values << 1;
    if (bigv <= 0) {
        gi.region0_count = 0;
        gi.region1_count = 0;
        gi.address1 = 0;
        gi.address2 = 0;
        gi.address3 = 0;
        gi.table_select = {0, 0, 0};
        return 0;
    }

    const auto& sfb = tables::scale_fact_band_index[s.mpeg.samplerate_index];

    // ---- Prefix tables over coefficient pairs ----------------------------
    // SFB boundaries and 2*big_values are always even, so every realisable
    // region edge falls on a pair boundary. One O(n_pairs) pass builds
    // everything the 128-combination region search needs in O(1) per segment.
    //
    //   SMALL_T   non-ESC tables and the max value they can encode
    //   pb_s[t]   prefix bits (incl. sign bits) for each small table
    //   pb16/24   prefix bits with ESC values clamped to 15 (family base)
    //   pesc      prefix count of escape values (>14) — linbits multiplier
    //   pmax      prefix max of |ix|
    static constexpr int SMALL_T[]    = {1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15};
    static constexpr int NUM_SMALL    = 13;

    const int n_pairs = bigv >> 1;
    auto& pb_s = s.subdivide_scratch.pb_s;
    auto& pb16 = s.subdivide_scratch.pb16;
    auto& pb24 = s.subdivide_scratch.pb24;
    auto& pesc = s.subdivide_scratch.pesc;
    auto& pmax = s.subdivide_scratch.pmax;


    const std::uint16_t* hl16 = huffman::at(16).hlen;
    const std::uint16_t* hl24 = huffman::at(24).hlen;
    const std::uint16_t* hl_s[NUM_SMALL];
    int xlen_s[NUM_SMALL];
    for (int k = 0; k < NUM_SMALL; ++k) {
        const auto& h = huffman::at(SMALL_T[k]);
        hl_s[k]   = h.hlen;
        xlen_s[k] = h.xlen;
    }

    pmax[0] = 0; pesc[0] = 0; pb16[0] = 0; pb24[0] = 0;
    for (int k = 0; k < NUM_SMALL; ++k) pb_s[k][0] = 0;

    static constexpr std::array<int, 16> START_K = {
        0, 0, 1, 3, 5, 5, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11
    };

    const std::uint32_t const_hl16_0 = hl16[0];
    const std::uint32_t const_hl24_0 = hl24[0];
    std::array<std::uint32_t, NUM_SMALL> const_hl_s_0{};
    for (int k = 0; k < NUM_SMALL; ++k) {
        const_hl_s_0[k] = hl_s[k][0];
    }

    for (int p = 0; p < n_pairs; ++p) {
        const int x = ix[2 * p], y = ix[2 * p + 1];
        if (x == 0 && y == 0) {
            pmax[p + 1] = pmax[p];
            pesc[p + 1] = pesc[p];
            pb16[p + 1] = pb16[p] + const_hl16_0;
            pb24[p + 1] = pb24[p] + const_hl24_0;
            for (int k = 0; k < NUM_SMALL; ++k) {
                pb_s[k][p + 1] = pb_s[k][p] + const_hl_s_0[k];
            }
            continue;
        }
        const int sign = (x ? 1 : 0) + (y ? 1 : 0);
        const int m    = (x > y) ? x : y;
        pmax[p + 1] = static_cast<std::uint16_t>((m > pmax[p]) ? m : pmax[p]);
        pesc[p + 1] = static_cast<std::uint16_t>(pesc[p] + (x > 14) + (y > 14));
        const int cx = (x > 14) ? 15 : x;
        const int cy = (y > 14) ? 15 : y;
        const int eidx = cx * 16 + cy;
        pb16[p + 1] = pb16[p] + static_cast<std::uint32_t>(hl16[eidx] + sign);
        pb24[p + 1] = pb24[p] + static_cast<std::uint32_t>(hl24[eidx] + sign);
        
        const int start_k = (m < 16) ? START_K[static_cast<std::size_t>(m)] : 13;
        for (int k = 0; k < start_k; ++k) {
            pb_s[k][p + 1] = pb_s[k][p] + 0x40000u;
        }
        for (int k = start_k; k < NUM_SMALL; ++k) {
            const int xl = xlen_s[k];
            pb_s[k][p + 1] = pb_s[k][p] + static_cast<std::uint32_t>(hl_s[k][x * xl + y] + sign);
        }
    }

    // Best (table, bits) for the pair range [a, b).
    struct Seg { int table; int bits; };
    auto seg_best = [&](int a, int b) -> Seg {
        if (b <= a) return {0, 0};
        // pmax is a running prefix max: if it grew inside [a,b) the segment
        // max equals pmax[b]; otherwise the true max sits before `a` and a
        // short scan recovers the segment-local value exactly.
        int seg_max = (pmax[b] > pmax[a]) ? pmax[b] : 0;
        if (seg_max == 0) {
            // Running max did not grow: scan (rare, short segments only when
            // the largest values sit before `a`).
            for (int p = a; p < b; ++p) {
                const int x = ix[2 * p], y = ix[2 * p + 1];
                if (x > seg_max) seg_max = x;
                if (y > seg_max) seg_max = y;
            }
        }
        if (seg_max == 0) return {0, 0};
        if (seg_max < 15) {
            int best_t = 0, best_b = INT_MAX;
            for (int k = 0; k < NUM_SMALL; ++k) {
                if (xlen_s[k] <= seg_max) continue;
                const int b_ = static_cast<int>(pb_s[k][b] - pb_s[k][a]);
                if (b_ < best_b) { best_b = b_; best_t = SMALL_T[k]; }
            }
            if (best_b < 0x40000) return {best_t, best_b};
            // (unreachable: tables 13/15 cover every value < 15)
        }
        // ESC path.
        const int esc = pesc[b] - pesc[a];
        const int lin_need = (seg_max > 15) ? seg_max - 15 : 0;
        int t1;
        if (lin_need <= 1)         t1 = 16;
        else if (lin_need <= 3)    t1 = 17;
        else if (lin_need <= 7)    t1 = 18;
        else if (lin_need <= 15)   t1 = 19;
        else if (lin_need <= 63)   t1 = 20;
        else if (lin_need <= 255)  t1 = 21;
        else if (lin_need <= 1023) t1 = 22;
        else                       t1 = 23;
        const int t2 = t1 + 8;
        const int b1 = static_cast<int>(pb16[b] - pb16[a]) + esc * huffman::at(t1).linbits;
        const int b2 = static_cast<int>(pb24[b] - pb24[a]) + esc * huffman::at(t2).linbits;
        return (b2 < b1) ? Seg{t2, b2} : Seg{t1, b1};
    };

    // ---- Exhaustive (region0_count, region1_count) search ----------------
    constexpr int NUM_EDGES = 23;
    int bnd[NUM_EDGES];
    for (int k = 0; k < NUM_EDGES; ++k) {
        const int v = sfb[k];
        bnd[k] = ((v < bigv) ? v : bigv) >> 1;   // pair index, clamped
    }

    Seg r0_seg[17];
    for (int i1 = 1; i1 <= 16; ++i1) r0_seg[i1] = seg_best(0, bnd[i1]);
    // region-2 best for every possible i2 (start edge index)
    Seg r2_seg[NUM_EDGES];
    for (int i2 = 2; i2 < NUM_EDGES; ++i2) r2_seg[i2] = seg_best(bnd[i2], n_pairs);

    int best_bits = INT_MAX, best_r0 = 0, best_r1 = 0;
    for (int r0 = 0; r0 < 16; ++r0) {
        const int i1 = r0 + 1;
        const int b0 = r0_seg[i1].bits;
        for (int r1 = 0; r1 < 8; ++r1) {
            const int i2 = i1 + r1 + 1;
            if (i2 >= NUM_EDGES) break;
            int bits = b0 + r2_seg[i2].bits;
            if (bnd[i2] > bnd[i1]) {
                bits += seg_best(bnd[i1], bnd[i2]).bits;
            }
            if (bits < best_bits) { best_bits = bits; best_r0 = r0; best_r1 = r1; }
        }
        if (bnd[i1] >= n_pairs) break;   // region 0 already swallows everything
    }

    const int i1 = best_r0 + 1;
    const int i2 = best_r0 + best_r1 + 2;
    gi.region0_count = best_r0;
    gi.region1_count = best_r1;
    gi.address1      = bnd[i1] << 1;
    gi.address2      = bnd[i2] << 1;
    gi.address3      = bigv;
    gi.table_select  = {0, 0, 0};
    if (gi.address1 > 0)           gi.table_select[0] = r0_seg[i1].table;
    if (gi.address2 > gi.address1) gi.table_select[1] = seg_best(bnd[i1], bnd[i2]).table;
    if (bigv > gi.address2)        gi.table_select[2] = r2_seg[i2].table;
    return (best_bits == INT_MAX) ? 0 : best_bits;
}

int bin_search_StepSize(int desired_rate, int* ix, GrInfo& gi, EncoderState& s, int gr, int ch) {
    int next_val = -120, count = 120;
    while (count > 1) {
        int half = count / 2;
        int bit;
        if (quantize(ix, next_val + half, s, gr, ch) > 8192) {
            bit = 100000;
        } else {
            calc_runlen(ix, gi);
            bit  = count1_bitcount(ix, gi);
            subdivide(gi, s);
            bigv_tab_select(ix, gi);
            bit += bigv_bitcount(ix, gi);
        }
        if (bit < desired_rate) count = half;
        else { next_val += half; count -= half; }
    }
    return next_val;
}

int inner_loop(int* ix, int max_bits, GrInfo& gi, EncoderState& s, int gr, int ch) {
    if (max_bits < 0) gi.quantizerStepSize -= 1;
    int bits = 0;
    while (true) {
        gi.quantizerStepSize += 1;
        if (gi.quantizerStepSize > 128) {
            gi.quantizerStepSize = 128;
            std::fill_n(ix, GRANULE_SAMPLES, 0);
            bits = 0;
            break;
        }
        while (quantize(ix, gi.quantizerStepSize, s, gr, ch) > 8192) {
            gi.quantizerStepSize += 1;
            if (gi.quantizerStepSize > 128) break;
        }
        if (gi.quantizerStepSize > 128) {
            gi.quantizerStepSize = 128;
            std::fill_n(ix, GRANULE_SAMPLES, 0);
            bits = 0;
            break;
        }
        calc_runlen(ix, gi);
        bits  = count1_bitcount(ix, gi);
        subdivide(gi, s);
        bigv_tab_select(ix, gi);
        bits += bigv_bitcount(ix, gi);
        if (bits <= max_bits) break;
    }
    return bits;
}

int part2_length(int gr, int ch, EncoderState& s) {
    auto& gi    = s.side_info.gr[gr].ch[ch].tt;
    const int slen1 = tables::slen1_tab[static_cast<std::size_t>(gi.scalefac_compress)];
    const int slen2 = tables::slen2_tab[static_cast<std::size_t>(gi.scalefac_compress)];
    int bits = 0;
    const auto& scfsi = s.side_info.scfsi[ch];
    if (gr == 0 || !scfsi[0]) bits += 6 * slen1;
    if (gr == 0 || !scfsi[1]) bits += 5 * slen1;
    if (gr == 0 || !scfsi[2]) bits += 5 * slen2;
    if (gr == 0 || !scfsi[3]) bits += 5 * slen2;
    return bits;
}

int outer_loop(int max_bits, int* ix, int gr, int ch, EncoderState& s) {
    auto& gi = s.side_info.gr[gr].ch[ch].tt;
    gi.quantizerStepSize = bin_search_StepSize(max_bits, ix, gi, s, gr, ch);
    gi.part2_length      = part2_length(gr, ch, s);
    const int huff_bits  = max_bits - gi.part2_length;
    (void)inner_loop(ix, huff_bits, gi, s, gr, ch);
    // Finalise with the RD-optimal region split. This recount also repairs
    // the run-length fields in the all-zero starvation path (inner_loop can
    // zero ix while gi still carries counts from a previous iteration).
    calc_runlen(ix, gi);
    int bits  = count1_bitcount(ix, gi);
    bits     += best_subdivide_rd(ix, gi, s);
    gi.part2_3_length = gi.part2_length + bits;
    return gi.part2_3_length;
}

} // namespace rmps::quant
