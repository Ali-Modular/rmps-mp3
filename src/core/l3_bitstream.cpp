#include "core/l3_bitstream.hpp"
#include "core/encoder_state.hpp"
#include "core/bitstream.hpp"
#include "core/huffman_tables.hpp"
#include "quant/quantize.hpp"
#include <algorithm>
#include "core/tables.hpp"

#include <cassert>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace rmps::l3 {

namespace {

inline void abs_and_sign(int v, int& mag, int& sign) {
    if (v >= 0) { mag = v; sign = 0; }
    else        { mag = -v; sign = 1; }
}

void huffman_code(Bitstream& bs, int table_select, int x, int y) {
    int sx, sy;
    abs_and_sign(x, x, sx);
    abs_and_sign(y, y, sy);
    const auto& h = huffman::at(table_select);

    if (table_select > 15) {
        int linbits_x = 0, linbits_y = 0;
        if (x > 14) { linbits_x = x - 15; x = 15; }
        if (y > 14) { linbits_y = y - 15; y = 15; }
        const int idx   = x * h.ylen + y;
        const std::uint32_t code = h.code[idx];
        const int          cbits = h.hlen[idx];

        std::uint32_t ext  = 0;
        int           xb   = 0;
        if (x > 14) { ext |= static_cast<std::uint32_t>(linbits_x); xb += h.linbits; }
        if (x != 0) { ext = (ext << 1) | static_cast<std::uint32_t>(sx); xb += 1; }
        if (y > 14) {
            ext = (ext << h.linbits) | static_cast<std::uint32_t>(linbits_y);
            xb += h.linbits;
        }
        if (y != 0) { ext = (ext << 1) | static_cast<std::uint32_t>(sy); xb += 1; }

        bs.putbits(code, cbits);
        bs.putbits(ext,  xb);
    } else {
        const int idx = x * h.ylen + y;
        std::uint32_t code  = h.code[idx];
        int           cbits = h.hlen[idx];
        if (x != 0) { code = (code << 1) | static_cast<std::uint32_t>(sx); cbits += 1; }
        if (y != 0) { code = (code << 1) | static_cast<std::uint32_t>(sy); cbits += 1; }
        bs.putbits(code, cbits);
    }
}

void huffman_coder_count1(Bitstream& bs, const huffman::Table& h,
                          int v, int w, int x, int y) {
    int sv, sw, sx, sy;
    abs_and_sign(v, v, sv);
    abs_and_sign(w, w, sw);
    abs_and_sign(x, x, sx);
    abs_and_sign(y, y, sy);

    const int p = v + (w << 1) + (x << 2) + (y << 3);
    bs.putbits(h.code[p], h.hlen[p]);

    std::uint32_t code = 0;
    int           cb   = 0;
    if (v != 0) { code = static_cast<std::uint32_t>(sv); cb = 1; }
    if (w != 0) { code = (code << 1) | static_cast<std::uint32_t>(sw); cb += 1; }
    if (x != 0) { code = (code << 1) | static_cast<std::uint32_t>(sx); cb += 1; }
    if (y != 0) { code = (code << 1) | static_cast<std::uint32_t>(sy); cb += 1; }
    bs.putbits(code, cb);
}

void huffman_code_bits(EncoderState& s, Bitstream& bs, int* ix, GrInfo& gi) {
    const auto& sfb = tables::scale_fact_band_index[s.mpeg.samplerate_index];

    const std::size_t start_bits = bs.bit_count();
    const int bigvalues = gi.big_values << 1;

    int region1Start, region2Start;
    if (gi.window_switching_flag && gi.block_type == BLOCK_SHORT) {
        // Short blocks: fixed region split at coefficient 36; no region2.
        region1Start = 36;
        region2Start = bigvalues;
    } else {
        int scalefac_index = gi.region0_count + 1;
        region1Start   = sfb[scalefac_index];
        scalefac_index += gi.region1_count + 1;
        region2Start   = sfb[scalefac_index];
    }

    for (int i = 0; i < bigvalues; i += 2) {
        const int region = (i >= region1Start ? 1 : 0) + (i >= region2Start ? 1 : 0);
        const int t = gi.table_select[region];
        if (t) huffman_code(bs, t, ix[i], ix[i + 1]);
    }

    const std::size_t bigv_end_bits = bs.bit_count();
    // Count1 region (quadruples).
    const auto& h = huffman::at(gi.count1table_select + 32);
    const int count1End = bigvalues + (gi.count1 << 2);
    for (int i = bigvalues; i < count1End; i += 4) {
        huffman_coder_count1(bs, h, ix[i], ix[i + 1], ix[i + 2], ix[i + 3]);
    }

    // Stuffing bits (if part2_3_length declared more than actually written).
    std::int64_t bits = static_cast<std::int64_t>(bs.bit_count())
                      - static_cast<std::int64_t>(start_bits);
    std::int64_t want = gi.part2_3_length - gi.part2_length - bits;
    // Overrun: the Huffman data came out longer than the side info declares.
    // Nothing downstream can repair that - the decoder reads exactly
    // part2_3_length bits for this granule, so it desynchronises and every
    // following granule in the reservoir is read from the wrong place
    // (libmpg123: "dequantization failed"). The stuffing branch below only ever
    // handled the other direction, so this used to pass silently. Always on:
    // it prints nothing on a valid stream.
    if (want < 0) {
      int max_bv = 0, c1_big = 0, tail_nz = 0;
      for (int i = 0; i < bigvalues && i < 576; ++i)
        max_bv = std::max(max_bv, std::abs(ix[i]));
      for (int i = bigvalues; i < count1End && i < 576; ++i)
        if (std::abs(ix[i]) > 1) ++c1_big;
      for (int i = count1End; i < 576; ++i)
        if (ix[i] != 0) ++tail_nz;
      // Re-count the big-values region with the quantizer's own counter, at
      // the writer's region boundaries and (if present) at the stored
      // addresses. Counted == written but != declared means the declared
      // length went stale after counting; counted != written means the two
      // disagree about where the regions are.
      // ix is signed here (huffman_code strips the sign), count_bit wants
      // magnitudes: an earlier version of this recount fed it signed values
      // and read hlen[] at negative indices.
      std::array<int, 576> mag{};
      for (int i = 0; i < 576; ++i)
        mag[static_cast<std::size_t>(i)] = std::abs(ix[i]);
      // What the region splitter charges: the family base's lengths (16 or
      // 24) plus this table's linbits per escape value.
      const auto fam = [&](int a0, int a1, int t) {
        a0 = std::clamp(a0, 0, bigvalues);
        a1 = std::clamp(a1, 0, bigvalues);
        if (a1 <= a0 || t <= 15)
          return a1 > a0 ? rmps::quant::count_bit(mag.data(), a0, a1, t) : 0;
        const int base = (t >= 24) ? 24 : 16;
        int esc = 0;
        for (int i = a0; i < a1; ++i)
          esc += mag[static_cast<std::size_t>(i)] > 14;
        return rmps::quant::count_bit(mag.data(), a0, a1, base) +
               esc * (huffman::at(t).linbits - huffman::at(base).linbits);
      };
      const auto rc = [&](int a0, int a1, int t) {
        a0 = std::clamp(a0, 0, bigvalues);
        a1 = std::clamp(a1, 0, bigvalues);
        return a1 > a0 ? rmps::quant::count_bit(mag.data(), a0, a1, t) : 0;
      };
      const int w0 = rc(0, region1Start, gi.table_select[0]);
      const int w1 = rc(region1Start, region2Start, gi.table_select[1]);
      const int w2 = rc(region2Start, bigvalues, gi.table_select[2]);
      const long long bigv_written =
          static_cast<long long>(bigv_end_bits - start_bits);
      int mx[3] = {0, 0, 0};
      for (int i = 0; i < bigvalues && i < 576; ++i) {
        const int rg = (i >= region1Start ? 1 : 0) + (i >= region2Start ? 1 : 0);
        mx[rg] = std::max(mx[rg], std::abs(ix[i]));
      }
      std::fprintf(stderr,
                   "[bs]   regions at writer bounds: r1start=%d r2start=%d "
                   "counted %d+%d+%d=%d  bigv written=%lld  count1 written=%lld\n",
                   region1Start, region2Start, w0, w1, w2, w0 + w1 + w2,
                   bigv_written, static_cast<long long>(bits) - bigv_written);
      std::fprintf(stderr,
                   "[bs]   family-priced (what the splitter charges): "
                   "%d+%d+%d=%d\n",
                   fam(0, region1Start, gi.table_select[0]),
                   fam(region1Start, region2Start, gi.table_select[1]),
                   fam(region2Start, bigvalues, gi.table_select[2]),
                   fam(0, region1Start, gi.table_select[0]) +
                       fam(region1Start, region2Start, gi.table_select[1]) +
                       fam(region2Start, bigvalues, gi.table_select[2]));
      std::fprintf(stderr,
                   "[bs]   max|ix| per region %d/%d/%d  table linbits %d/%d/%d\n",
                   mx[0], mx[1], mx[2],
                   gi.table_select[0] > 15 ? huffman::at(gi.table_select[0]).linbits : 0,
                   gi.table_select[1] > 15 ? huffman::at(gi.table_select[1]).linbits : 0,
                   gi.table_select[2] > 15 ? huffman::at(gi.table_select[2]).linbits : 0);
      {
        const int a0 = rc(0, gi.address1, gi.table_select[0]);
        const int a1 = rc(gi.address1, gi.address2, gi.table_select[1]);
        const int a2 = rc(gi.address2, bigvalues, gi.table_select[2]);
        std::fprintf(stderr,
                     "[bs]   regions at stored addresses: a1=%d a2=%d a3=%d "
                     "counted %d+%d+%d=%d\n",
                     gi.address1, gi.address2, gi.address3, a0, a1, a2,
                     a0 + a1 + a2);
      }
      std::fprintf(stderr,
                   "[bs] OVERRUN frame=%lld declared=%d part2=%d written=%lld "
                   "over=%lld bt=%d ws=%d bigv=%d count1=%d tsel=%d,%d,%d "
                   "r0=%d r1=%d c1tab=%d max|ix|bigv=%d |ix|>1_in_count1=%d "
                   "nonzero_after_count1=%d\n",
                   static_cast<long long>(s.dbg_frame_no), gi.part2_3_length,
                   gi.part2_length, static_cast<long long>(bits),
                   static_cast<long long>(-want), gi.block_type,
                   gi.window_switching_flag, gi.big_values, gi.count1,
                   gi.table_select[0], gi.table_select[1], gi.table_select[2],
                   gi.region0_count, gi.region1_count, gi.count1table_select,
                   max_bv, c1_big, tail_nz);
    }
    if (want > 0) {
        const int stuffingWords = static_cast<int>(want / 32);
        const int remaining     = static_cast<int>(want % 32);
        for (int i = 0; i < stuffingWords; ++i) bs.putbits(0xFFFFFFFFu, 32);
        if (remaining > 0) bs.putbits((1u << remaining) - 1u, remaining);
    }
}

void encode_side_info(EncoderState& s, Bitstream& bs, int main_data_begin) {
    auto& si   = s.side_info;
    auto& mpeg = s.mpeg;

    bs.putbits(0x7ff, 11);
    bs.putbits(static_cast<std::uint32_t>(mpeg.version), 2);
    bs.putbits(static_cast<std::uint32_t>(mpeg.layer),   2);
    bs.putbits(mpeg.crc ? 0u : 1u, 1);
    bs.putbits(static_cast<std::uint32_t>(mpeg.bitrate_index), 4);
    bs.putbits(static_cast<std::uint32_t>(mpeg.samplerate_index % 3), 2);
    bs.putbits(static_cast<std::uint32_t>(mpeg.padding), 1);
    bs.putbits(static_cast<std::uint32_t>(mpeg.ext), 1);
    bs.putbits(static_cast<std::uint32_t>(mpeg.mode), 2);
    bs.putbits(static_cast<std::uint32_t>(mpeg.mode_ext), 2);
    bs.putbits(mpeg.copyright ? 1u : 0u, 1);
    bs.putbits(mpeg.original  ? 1u : 0u, 1);
    bs.putbits(static_cast<std::uint32_t>(mpeg.emph), 2);

    if (mpeg.version == 3) {
        bs.putbits(static_cast<std::uint32_t>(main_data_begin), 9);
        if (s.wave.channels == 2) bs.putbits(static_cast<std::uint32_t>(si.private_bits), 3);
        else                      bs.putbits(static_cast<std::uint32_t>(si.private_bits), 5);
    } else {
        bs.putbits(static_cast<std::uint32_t>(main_data_begin), 8);
        if (s.wave.channels == 2) bs.putbits(static_cast<std::uint32_t>(si.private_bits), 2);
        else                      bs.putbits(static_cast<std::uint32_t>(si.private_bits), 1);
    }

    if (mpeg.version == 3) {
        for (int ch = 0; ch < s.wave.channels; ++ch)
            for (int b = 0; b < 4; ++b)
                bs.putbits(static_cast<std::uint32_t>(si.scfsi[ch][b]), 1);
    }

    for (int gr = 0; gr < mpeg.granules_per_frame; ++gr) {
        for (int ch = 0; ch < s.wave.channels; ++ch) {
            auto& gi = si.gr[gr].ch[ch].tt;
            bs.putbits(static_cast<std::uint32_t>(gi.part2_3_length),     12);
            bs.putbits(static_cast<std::uint32_t>(gi.big_values),         9);
            if (std::getenv("RMPS_SIDE_CHECK")) {
                if (gi.global_gain < 0 || gi.global_gain > 255)
                    std::fprintf(stderr,
                                 "[side] global_gain=%d OUT OF RANGE (8-bit "
                                 "field) -> transmits %d\n",
                                 gi.global_gain, gi.global_gain & 255);
                if (gi.part2_3_length > 4095)
                    std::fprintf(stderr, "[side] part2_3_length=%d > 4095\n",
                                 gi.part2_3_length);
                if (gi.big_values > 288)
                    std::fprintf(stderr, "[side] big_values=%d > 288\n",
                                 gi.big_values);
            }
            bs.putbits(static_cast<std::uint32_t>(gi.global_gain),        8);
            if (mpeg.version == 3) bs.putbits(static_cast<std::uint32_t>(gi.scalefac_compress), 4);
            else                   bs.putbits(static_cast<std::uint32_t>(gi.scalefac_compress), 9);

            bs.putbits(static_cast<std::uint32_t>(gi.window_switching_flag), 1);
            if (gi.window_switching_flag) {
                bs.putbits(static_cast<std::uint32_t>(gi.block_type),       2);
                bs.putbits(static_cast<std::uint32_t>(gi.mixed_block_flag), 1);
                // Only two table_select entries for window-switched blocks.
                for (int region = 0; region < 2; ++region)
                    bs.putbits(static_cast<std::uint32_t>(gi.table_select[region]), 5);
                for (int w = 0; w < 3; ++w)
                    bs.putbits(static_cast<std::uint32_t>(gi.subblock_gain[w]), 3);
            } else {
                for (int region = 0; region < 3; ++region)
                    bs.putbits(static_cast<std::uint32_t>(gi.table_select[region]), 5);
                bs.putbits(static_cast<std::uint32_t>(gi.region0_count), 4);
                bs.putbits(static_cast<std::uint32_t>(gi.region1_count), 3);
            }

            if (mpeg.version == 3) bs.putbits(static_cast<std::uint32_t>(gi.preflag), 1);
            bs.putbits(static_cast<std::uint32_t>(gi.scalefac_scale),     1);
            bs.putbits(static_cast<std::uint32_t>(gi.count1table_select), 1);
        }
    }
}

void encode_main_data(EncoderState& s, Bitstream& md) {
    auto& si = s.side_info;

    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
        for (int ch = 0; ch < s.wave.channels; ++ch) {
            auto& gi = si.gr[gr].ch[ch].tt;
            const int slen1 = tables::slen1_tab[static_cast<std::size_t>(gi.scalefac_compress)];
            const int slen2 = tables::slen2_tab[static_cast<std::size_t>(gi.scalefac_compress)];
            auto& ix  = s.l3_enc[ch][gr];
            auto& scf = si.scfsi[ch];
            const std::size_t sf_start = md.bit_count();

            if (gi.window_switching_flag && gi.block_type == BLOCK_SHORT) {
                // Short block: per (sfb, window) scalefactors; no scfsi.
                // sfb 0..5 use slen1, sfb 6..11 use slen2 (sfb 12 is implicit 0).
                for (int sfb = 0; sfb < 6; ++sfb)
                    for (int w = 0; w < NUM_WINDOWS; ++w)
                        md.putbits(static_cast<std::uint32_t>(s.scalefactor.s[gr][ch][w][sfb]), slen1);
                for (int sfb = 6; sfb < 12; ++sfb)
                    for (int w = 0; w < NUM_WINDOWS; ++w)
                        md.putbits(static_cast<std::uint32_t>(s.scalefactor.s[gr][ch][w][sfb]), slen2);
            } else {
                if (gr == 0 || scf[0] == 0)
                    for (int sfb = 0; sfb < 6;  ++sfb) md.putbits(static_cast<std::uint32_t>(s.scalefactor.l[gr][ch][sfb]), slen1);
                if (gr == 0 || scf[1] == 0)
                    for (int sfb = 6; sfb < 11; ++sfb) md.putbits(static_cast<std::uint32_t>(s.scalefactor.l[gr][ch][sfb]), slen1);
                if (gr == 0 || scf[2] == 0)
                    for (int sfb = 11; sfb < 16;++sfb) md.putbits(static_cast<std::uint32_t>(s.scalefactor.l[gr][ch][sfb]), slen2);
                if (gr == 0 || scf[3] == 0)
                    for (int sfb = 16; sfb < 21;++sfb) md.putbits(static_cast<std::uint32_t>(s.scalefactor.l[gr][ch][sfb]), slen2);
            }

            const int sf_bits = static_cast<int>(md.bit_count() - sf_start);
            if (sf_bits != gi.part2_length) {
                static int p2 = 0;
                if (p2 < 20) {
                    ++p2;
                    std::fprintf(stderr,
                                 "rmps: part2_length mismatch gr=%d ch=%d "
                                 "declared=%d written=%d (delta %+d)\n",
                                 gr, ch, gi.part2_length, sf_bits,
                                 sf_bits - gi.part2_length);
                }
            }
            if (s.dbg_fin_valid[gr][ch]) {
              s.dbg_fin_valid[gr][ch] = false;
              const auto &snap = s.dbg_fin_ix[gr][ch];
              int ndiff = 0;
              char buf[512];
              int off = 0;
              for (int i = 0; i < GRANULE_SAMPLES; ++i) {
                const int now = std::abs(ix[static_cast<std::size_t>(i)]);
                const int was = snap[static_cast<std::size_t>(i)];
                if (now != was) {
                  if (ndiff < 12 && off < 440)
                    off += std::snprintf(buf + off, sizeof(buf) - off,
                                         " [%d]%d->%d", i, was, now);
                  ++ndiff;
                }
              }
              // A *rise* in part2_3_length is reservoir drain stuffing and
              // harmless; only changed data or run lengths, or a drop, mean
              // the side info no longer describes what is written.
              if (ndiff || gi.big_values != s.dbg_fin_bigv[gr][ch] ||
                  gi.count1 != s.dbg_fin_count1[gr][ch] ||
                  gi.part2_3_length < s.dbg_fin_p23[gr][ch])
                std::fprintf(stderr,
                             "[bs] CHANGED-AFTER-COUNT frame=%lld gr=%d ch=%d "
                             "bigv %d->%d count1 %d->%d p23 %d->%d lines=%d%s\n",
                             static_cast<long long>(s.dbg_frame_no), gr, ch,
                             s.dbg_fin_bigv[gr][ch], gi.big_values,
                             s.dbg_fin_count1[gr][ch], gi.count1,
                             s.dbg_fin_p23[gr][ch], gi.part2_3_length, ndiff,
                             ndiff ? buf : "");
            }
            huffman_code_bits(s, md, ix.data(), gi);
            const int wrote = static_cast<int>(md.bit_count() - sf_start);
            if (wrote != gi.part2_3_length) {
                static int reported = 0;
                if (reported < 20) {
                    ++reported;
                    std::fprintf(stderr,
                                 "rmps: part2_3_length mismatch gr=%d ch=%d "
                                 "declared=%d written=%d (delta %+d)\n",
                                 gr, ch, gi.part2_3_length, wrote,
                                 wrote - gi.part2_3_length);
                }
            }
        }
    }
}

} // namespace

void format_bitstream(EncoderState& s) {
    // Restore signs from original MDCT spectrum.
    for (int ch = 0; ch < s.wave.channels; ++ch) {
        for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
            auto& pi = s.l3_enc[ch][gr];
            auto& pr = s.mdct_freq[ch][gr];
            for (int i = 0; i < GRANULE_SAMPLES; ++i) {
                if (pr[i] < 0 && pi[i] > 0) pi[i] = -pi[i];
            }
        }
    }

    // ---- Encoder-model vs decoder-model check (RMPS_DEQUANT_CHECK) ----
    //
    // Every internal metric in this encoder measures its own reconstruction.
    // If the encoder's dequantization convention differs from the decoder's,
    // all of them agree with each other and all of them are wrong, which is
    // exactly the state a 36-44 dB internal SDR alongside a far worse decode
    // describes. So compare the two formulas directly on the emitted values:
    //
    //   encoder : pow43[ix] * dequant_factor        (its own scale chain)
    //   decoder : ix^(4/3) * 2^((global_gain-210)/4)
    //                      * 2^(-0.5*(1+scalefac_scale)*(scalefac + preflag*pretab))
    //
    // Their ratio should be a single constant across every coefficient. Any
    // *variation* in the ratio is a convention mismatch and localises it.
    if (std::getenv("RMPS_DEQUANT_CHECK")) {
        static int frames_left = 3;
        if (frames_left > 0) {
            --frames_left;
            const auto& bt = tables::scale_fact_band_index[s.mpeg.samplerate_index];
            for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
                const int ch = 0;
                const auto& gi = s.side_info.gr[gr].ch[ch].tt;
                if (gi.window_switching_flag && gi.block_type == BLOCK_SHORT)
                    continue;
                const auto& ix = s.l3_enc[ch][gr];
                const auto& xr = s.mdct_freq[ch][gr];
                // the loudest few coefficients of this granule
                int best[4] = {-1, -1, -1, -1};
                for (int i = 0; i < GRANULE_SAMPLES; ++i) {
                    for (int k = 0; k < 4; ++k) {
                        if (best[k] < 0 || std::abs(ix[i]) > std::abs(ix[best[k]])) {
                            for (int j = 3; j > k; --j) best[j] = best[j-1];
                            best[k] = i;
                            break;
                        }
                    }
                }
                std::fprintf(stderr,
                             "[deq] gr=%d gg=%d scale=%d preflag=%d\n",
                             gr, gi.global_gain, gi.scalefac_scale, gi.preflag);
                for (int k = 0; k < 4; ++k) {
                    const int i = best[k];
                    if (i < 0 || ix[i] == 0) continue;
                    int sfb = 0;
                    while (sfb < 21 && i >= bt[sfb + 1]) ++sfb;
                    const int sf = s.scalefactor.l[gr][ch][sfb];
                    const int pre = gi.preflag ? tables::pretab[sfb] : 0;
                    const double dec =
                        std::pow(std::abs((double)ix[i]), 4.0/3.0)
                        * std::pow(2.0, (gi.global_gain - 210) / 4.0)
                        * std::pow(2.0, -0.5 * (1 + gi.scalefac_scale) * (sf + pre));
                    std::fprintf(stderr,
                                 "      bin=%3d sfb=%2d xr=%12d ix=%5d sf=%2d "
                                 "decoder=%.6g  xr/decoder=%.6g\n",
                                 i, sfb, xr[i], ix[i], sf, dec,
                                 std::abs((double)xr[i]) / (dec > 0 ? dec : 1));
                }
            }
        }
    }

    // ---- Cross-frame bit-reservoir emission (main_data_begin) ----
    // Each frame contributes a fixed slot area (bits_per_frame minus
    // header+side info) to a shared main_data byte stream. A frame whose
    // main_data is shorter than its slots leaves the remainder to FUTURE
    // frames, which reach back into it via main_data_begin. This converts
    // the stuffing that quiet frames used to emit into bit savings an
    // attack frame can draw down (reservoir::frame_end caps the bank at
    // ResvMax and keeps it byte-aligned).
    //
    // Accounting is done here in BYTES from the actually-produced main
    // data, never from the planner's ResvSize, so the emitted
    // main_data_begin can not drift from the true stream layout (the
    // failure mode that killed the earlier reservoir attempt).
    const int slot_bytes =
        s.mpeg.bits_per_frame / 8 - s.sideinfo_len / 8;

    // resv_bytes = unfilled slot bytes of already-produced frames; that is
    // exactly where this frame's main_data starts, i.e. main_data_begin.
    if (s.resv_bytes < 0 || s.resv_bytes > 511) {
        std::fprintf(stderr,
                     "rmps: reservoir accounting out of range (%d bytes) — "
                     "clamping; stream may glitch\n", s.resv_bytes);
        s.resv_bytes = s.resv_bytes < 0 ? 0 : 511;
    }
    Bitstream head(128);
    encode_side_info(s, head, /*main_data_begin=*/s.resv_bytes);
    head.flush_bytes();          // header+side info is a whole byte count

    Bitstream md(8192);
    encode_main_data(s, md);
    if (md.bit_count() % 8 != 0) {
        // reservoir::frame_end keeps every frame's main_data byte-aligned;
        // reaching this means a part2_3_length accounting bug upstream.
        std::fprintf(stderr, "rmps: main_data not byte-aligned (%zu bits)\n",
                     md.bit_count());
        md.putbits(0, static_cast<int>(8 - md.bit_count() % 8));
    }
    md.flush_bytes();
    const int md_bytes = static_cast<int>(md.byte_position());

    s.pending_frames.push_back(
        {std::vector<std::uint8_t>(head.data(),
                                   head.data() + head.byte_position()),
         slot_bytes});
    s.md_fifo.insert(s.md_fifo.end(), md.data(), md.data() + md.byte_position());
    s.resv_bytes += slot_bytes - md_bytes;

    // Emit every pending frame whose slot area the fifo can already fill.
    // A frame that banked bits waits until later frames produce the main
    // data that occupies its tail slots.
    while (!s.pending_frames.empty()
           && s.md_fifo.size() >=
                  static_cast<std::size_t>(s.pending_frames.front().slot)) {
        const auto& pf = s.pending_frames.front();
        for (const auto b : pf.head) s.bs.putbits(b, 8);
        for (int i = 0; i < pf.slot; ++i)
            s.bs.putbits(s.md_fifo[static_cast<std::size_t>(i)], 8);
        s.md_fifo.erase(s.md_fifo.begin(), s.md_fifo.begin() + pf.slot);
        s.pending_frames.erase(s.pending_frames.begin());
    }
    s.bs.flush_bytes();
}

void flush_bitstream(EncoderState& s) {
    // Stream end: no more main data is coming, so zero-pad the fifo far
    // enough to serve every still-pending frame's slot area.
    while (!s.pending_frames.empty()) {
        const auto pf = s.pending_frames.front();
        s.pending_frames.erase(s.pending_frames.begin());
        if (s.md_fifo.size() < static_cast<std::size_t>(pf.slot))
            s.md_fifo.resize(static_cast<std::size_t>(pf.slot), 0u);
        for (const auto b : pf.head) s.bs.putbits(b, 8);
        for (int i = 0; i < pf.slot; ++i)
            s.bs.putbits(s.md_fifo[static_cast<std::size_t>(i)], 8);
        s.md_fifo.erase(s.md_fifo.begin(), s.md_fifo.begin() + pf.slot);
    }
    s.bs.flush_bytes();
    s.resv_bytes = 0;
}

} // namespace rmps::l3
