#include "quant/quantize.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include "core/huffman_tables.hpp"
#include "core/tables.hpp"

#include <algorithm>
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
    for (int i = 0; i < 128; ++i) {
        s.steptab[i] = std::pow(2.0, (127.0 - i) / 4.0);
        const double v = s.steptab[i] * 2.0;
        s.steptabi[i] = (v > 0x7fffffff)
            ? 0x7fffffff
            : static_cast<std::int32_t>(v + 0.5);
    }
    for (int i = 0; i < 10000; ++i) {
        s.int2idx[i] = static_cast<int>(std::pow(static_cast<double>(i), 0.75) - 0.0946 + 0.5);
    }
}

int quantize(int* ix, int stepsize, EncoderState& s) {
    const std::int32_t scalei = s.steptabi[stepsize + 127];
    if (fx::mulr(s.xrmax, scalei) > 165140) return 16384;

    int max_val = 0;
    const double step = s.steptab[stepsize + 127];
    for (int i = 0; i < GRANULE_SAMPLES; ++i) {
        const std::int32_t x = s.xrabs[i];
        const std::int32_t ln = fx::mulr(x, scalei);
        int v;
        if (ln < 10000) v = s.int2idx[ln];
        else {
            const double dbl = static_cast<double>(x) * step * 4.656612875e-10;
            v = static_cast<int>(std::pow(std::max(dbl, 0.0), 0.75));
        }
        ix[i] = v;
        if (v > max_val) max_val = v;
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
    int sum0 = 0, sum1 = 0;
    const int start_idx = gi.big_values << 1;
    const auto* h32 = huffman::at(32).hlen;
    const auto* h33 = huffman::at(33).hlen;
    for (int k = 0; k < gi.count1; ++k) {
        const int i = start_idx + k * 4;
        const int v = ix[i], w = ix[i+1], x = ix[i+2], y = ix[i+3];
        const int p = v + (w << 1) + (x << 2) + (y << 3);
        int signbits = 0;
        if (v) ++signbits;
        if (w) ++signbits;
        if (x) ++signbits;
        if (y) ++signbits;
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

int bin_search_StepSize(int desired_rate, int* ix, GrInfo& gi, EncoderState& s) {
    int next_val = -120, count = 120;
    while (count > 1) {
        int half = count / 2;
        int bit;
        if (quantize(ix, next_val + half, s) > 8192) {
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

int inner_loop(int* ix, int max_bits, GrInfo& gi, EncoderState& s) {
    if (max_bits < 0) gi.quantizerStepSize -= 1;
    int bits = 0;
    while (true) {
        gi.quantizerStepSize += 1;
        while (quantize(ix, gi.quantizerStepSize, s) > 8192) gi.quantizerStepSize += 1;
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
    gi.quantizerStepSize = bin_search_StepSize(max_bits, ix, gi, s);
    gi.part2_length      = part2_length(gr, ch, s);
    const int huff_bits  = max_bits - gi.part2_length;
    const int bits       = inner_loop(ix, huff_bits, gi, s);
    gi.part2_3_length    = gi.part2_length + bits;
    return gi.part2_3_length;
}

} // namespace rmps::quant
