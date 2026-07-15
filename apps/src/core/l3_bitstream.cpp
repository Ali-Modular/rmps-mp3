#include "core/l3_bitstream.hpp"
#include "core/encoder_state.hpp"
#include "core/bitstream.hpp"
#include "core/huffman_tables.hpp"
#include "core/tables.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>

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

            huffman_code_bits(s, md, ix.data(), gi);
        }
    }
}

} // namespace

namespace {

/// Emit every queued frame whose main_data slot the FIFO can fully cover.
/// Frames are emitted strictly in order; a frame whose slot is still waiting
/// for future main_data (its tail is reservoir space for upcoming frames)
/// stays queued.
void emit_ready_frames(EncoderState& s) {
    while (!s.pending_frames.empty()) {
        const auto& f = s.pending_frames.front();
        if (s.md_fifo.size() - s.md_emitted < static_cast<std::size_t>(f.slot))
            break;
        for (const std::uint8_t b : f.head)
            s.bs.putbits(b, 8);
        const std::uint8_t* src = s.md_fifo.data() + s.md_emitted;
        for (int i = 0; i < f.slot; ++i)
            s.bs.putbits(src[i], 8);
        s.bs.flush_bytes();
        s.md_emitted += static_cast<std::size_t>(f.slot);
        s.pending_frames.erase(s.pending_frames.begin());
    }
    // Compact the consumed FIFO prefix so it does not grow with the stream.
    if (s.md_emitted >= 4096) {
        s.md_fifo.erase(s.md_fifo.begin(),
                        s.md_fifo.begin() + static_cast<std::ptrdiff_t>(s.md_emitted));
        s.md_emitted = 0;
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

    // ----- Cross-frame bit reservoir (main_data_begin) -----
    // The FIFO byte counts below are the single source of truth for
    // main_data_begin: every frame contributes a fixed-size main_data slot
    // (capacity) and appends its actual main_data bytes (consumption); the
    // reservoir is exactly capacity-minus-consumption. The earlier
    // implementation derived main_data_begin from the parallel ResvSize
    // bookkeeping, whose integer-truncation drift corrupted the stream —
    // here ResvSize is re-synced FROM the byte counts instead.
    const int si_bytes    = s.sideinfo_len / 8;               // incl. 4-byte header
    const int frame_bytes = s.mpeg.bits_per_frame / 8;
    const int slot_bytes  = frame_bytes - si_bytes;

    // Header + side info, pointing main_data_begin at the reservoir depth
    // accumulated by the previous frames (0 for the first frame).
    const int mdb_cap = (s.mpeg.version == 3) ? 511 : 255;
    const int mdb     = std::min(s.resv_bytes, mdb_cap);
    Bitstream head(64);
    encode_side_info(s, head, mdb);
    head.flush_bytes();

    // Main data: scalefactors + huffman for all granules, then any reservoir
    // drain as ancillary stuffing bits. reservoir::frame_end keeps the total
    // byte-aligned; the trailing pad is defensive only.
    Bitstream md(4096);
    encode_main_data(s, md);
    int drain = s.side_info.resvDrain;
    s.side_info.resvDrain = 0;
    while (drain >= 32) { md.putbits(0u, 32); drain -= 32; }
    if (drain > 0) md.putbits(0u, drain);
    const int partial = static_cast<int>(md.bit_count() % 8);
    if (partial) md.putbits(0u, 8 - partial);
    md.flush_bytes();

    const int md_bytes = static_cast<int>(md.byte_position());
    s.md_fifo.insert(s.md_fifo.end(), md.data(), md.data() + md_bytes);
    EncoderState::PendingFrame pf;
    pf.head.assign(head.data(), head.data() + head.byte_position());
    pf.slot = slot_bytes;
    s.pending_frames.push_back(std::move(pf));

    // Advance the reservoir and re-sync the planner's view to ground truth.
    s.resv_bytes += slot_bytes - md_bytes;
    if (s.resv_bytes < 0) s.resv_bytes = 0;  // unreachable: budgets are capped
    if (s.resv_bytes > mdb_cap) s.resv_bytes = mdb_cap;
    s.ResvSize = 8 * s.resv_bytes;

    emit_ready_frames(s);
}

void flush_bitstream(EncoderState& s) {
    // Zero-pad the FIFO so every queued frame's slot is complete, then emit.
    std::size_t needed = 0;
    for (const auto& f : s.pending_frames)
        needed += static_cast<std::size_t>(f.slot);
    const std::size_t have = s.md_fifo.size() - s.md_emitted;
    if (needed > have)
        s.md_fifo.insert(s.md_fifo.end(), needed - have, std::uint8_t{0});
    emit_ready_frames(s);
    s.resv_bytes = 0;
    s.ResvSize   = 0;
}

} // namespace rmps::l3
