// MPEG-1 Layer III decoder — see include/rmps/decoder.hpp.
//
// A plain floating-point implementation of ISO/IEC 11172-3's decoding
// process, in the order the standard gives it: header and side info, the
// reservoir, scalefactors, Huffman, requantization, short-block reordering,
// M/S stereo, alias reduction, IMDCT with overlap-add, frequency inversion,
// polyphase synthesis. It reads the encoder's own Huffman and band tables, so
// encoder and decoder cannot disagree about either.
//
// The three bends (DecoderBends) sit at the points they name: `rot` on the
// main data as it enters the reservoir, `law` in requantization, `loop`
// between requantization and the filterbank.

#include "rmps/decoder.hpp"

#include "core/huffman_tables.hpp"
#include "core/tables.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace rmps {

namespace {

constexpr double PI = 3.14159265358979323846;

constexpr int kLines    = 576;
constexpr int kSubbands = 32;
constexpr int kMaxLoop  = 8;
/// Main data a frame can reach back for (main_data_begin is 9 bits) plus the
/// largest frame's own: 511 + 1441 bytes, rounded up.
constexpr int kResvBytes = 2048;

constexpr std::array<int, 15> kBitratesV1L3 = {
    0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320};
constexpr std::array<int, 3> kSampleratesV1 = {44100, 48000, 32000};
/// Index into tables::scale_fact_band_index for each of the above.
constexpr std::array<int, 3> kTableIndex = {0, 1, 2};

constexpr std::array<double, 8> kAliasCoef = {
    -0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037};

// ---- Huffman ----------------------------------------------------------------
// One binary tree per table, built from the encoder's code words. A child of
// 0 is a code no table contains, which only a damaged stream reaches; the
// walk stops there and reads a zero.
struct Tree {
    std::vector<std::array<std::int16_t, 2>> node;
};

struct Trees {
    std::array<Tree, huffman::NUM_TABLES> t;

    Trees() {
        for (int ti = 0; ti < huffman::NUM_TABLES; ++ti) {
            const auto& h = huffman::at(ti);
            auto& tree = t[ti];
            tree.node.push_back({0, 0});
            if (!h.code)
                continue;
            for (int v = 0; v < h.count; ++v) {
                const int len = h.hlen[v];
                if (len <= 0)
                    continue;
                insert(tree, h.code[v], len, v);
            }
        }
    }

    static void insert(Tree& tree, unsigned code, int len, int value) {
        int n = 0;
        for (int b = len - 1; b >= 0; --b) {
            const int bit = (code >> b) & 1;
            if (b == 0) {
                tree.node[n][bit] = static_cast<std::int16_t>(-(value + 1));
                return;
            }
            int c = tree.node[n][bit];
            if (c <= 0) {
                tree.node.push_back({0, 0});
                c = static_cast<int>(tree.node.size()) - 1;
                tree.node[n][bit] = static_cast<std::int16_t>(c);
            }
            n = c;
        }
    }
};

const Trees& trees() {
    static const Trees t;
    return t;
}

// ---- Bits -------------------------------------------------------------------
// Reads past the end as zeros, so a damaged length cannot run off the buffer.
struct BitReader {
    const std::uint8_t* p = nullptr;
    int len_bits = 0;
    int pos = 0;

    int bit() {
        if (pos >= len_bits || pos < 0) {
            ++pos;
            return 0;
        }
        const int b = (p[pos >> 3] >> (7 - (pos & 7))) & 1;
        ++pos;
        return b;
    }

    unsigned get(int n) {
        unsigned v = 0;
        for (int i = 0; i < n; ++i)
            v = (v << 1) | static_cast<unsigned>(bit());
        return v;
    }

    int huff(const Tree& tree) {
        int n = 0;
        for (int depth = 0; depth < 32; ++depth) {
            const int c = tree.node[n][bit()];
            if (c < 0)
                return -c - 1;
            if (c == 0)
                return 0;
            n = c;
        }
        return 0;
    }
};

// ---- Side info --------------------------------------------------------------
struct Granule {
    int part2_3_length = 0;
    int big_values = 0;
    int global_gain = 0;
    int scalefac_compress = 0;
    int window_switching = 0;
    int block_type = 0;
    int mixed = 0;
    std::array<int, 3> table_select{};
    std::array<int, 3> subblock_gain{};
    int region0_count = 0;
    int region1_count = 0;
    int preflag = 0;
    int scalefac_scale = 0;
    int count1table = 0;
};

struct Frame {
    int samplerate_index = 0;   ///< 0..2 into kSampleratesV1
    int channels = 2;
    int mode = 0;
    int mode_ext = 0;
    int bytes = 0;              ///< whole frame, header included
    int header_bytes = 4;       ///< 6 with CRC
    int side_bytes = 32;
    int main_data_begin = 0;
    std::array<std::array<int, 4>, 2> scfsi{};
    std::array<std::array<Granule, 2>, 2> gr{};   ///< [gr][ch]
};

/// Spectra between requantization and the filterbank — what the loop bend
/// replays.
struct Spectra {
    std::array<std::array<std::array<float, kLines>, 2>, 2> xr{};  ///< [gr][ch]
    std::array<std::array<int, 2>, 2> block_type{};                ///< [gr][ch]
};

} // namespace

struct Decoder::Impl {
    DecoderBends bends{};

    std::vector<std::uint8_t> in;       ///< bytes not yet framed
    std::vector<float> out;             ///< returned PCM

    std::array<std::uint8_t, kResvBytes> resv{};
    int resv_len = 0;

    int samplerate = 0;

    // Scalefactors of granule 0, which scfsi lets granule 1 reuse.
    std::array<std::array<int, 22>, 2> sf_long_gr0{};

    // Filterbank state.
    std::array<std::array<std::array<float, 18>, kSubbands>, 2> overlap{};
    std::array<std::array<float, 1024>, 2> v{};
    std::array<int, 2> v_off{};

    // Tables built once per decoder (cheap, and keeps Impl self-contained).
    std::array<float, 8207> pow43{};
    std::array<float, 8> cs{}, ca{};
    std::array<std::array<float, 36>, 4> win{};      ///< per block type
    std::array<std::array<float, 18>, 36> imdct_l{}; ///< [n][k]
    std::array<std::array<float, 6>, 12> imdct_s{};  ///< [n][k]
    std::array<std::array<float, 32>, 64> synth{};   ///< [i][k]
    std::array<float, 512> d_win{};
    std::array<std::array<int, kLines>, 3> short_reorder{}; ///< per sr index

    // Bends.
    std::uint32_t rng = 0x9E3779B9u;
    double rot_skip = -1.0;          ///< bits until the next flip; <0 = redraw
    std::array<Spectra, kMaxLoop> history{};
    int history_head = 0;            ///< next slot to write
    int history_count = 0;
    bool looping = false;
    int loop_start = 0, loop_len = 1, loop_pos = 0;

    Impl() {
        (void)trees();
        out.reserve(1152 * 2 * 16);
        in.reserve(8192);

        for (int i = 0; i < static_cast<int>(pow43.size()); ++i)
            pow43[i] = static_cast<float>(std::pow(static_cast<double>(i), 4.0 / 3.0));
        for (int i = 0; i < 8; ++i) {
            const double c = kAliasCoef[i];
            const double sq = std::sqrt(1.0 + c * c);
            cs[i] = static_cast<float>(1.0 / sq);
            ca[i] = static_cast<float>(c / sq);
        }
        for (int i = 0; i < 36; ++i) {
            const double l = std::sin(PI / 36.0 * (i + 0.5));
            win[0][i] = static_cast<float>(l);
            win[1][i] = static_cast<float>(i < 18 ? l
                                         : i < 24 ? 1.0
                                         : i < 30 ? std::sin(PI / 12.0 * (i - 18 + 0.5))
                                                  : 0.0);
            win[3][i] = static_cast<float>(i < 6  ? 0.0
                                         : i < 12 ? std::sin(PI / 12.0 * (i - 6 + 0.5))
                                         : i < 18 ? 1.0
                                                  : l);
            win[2][i] = i < 12 ? static_cast<float>(std::sin(PI / 12.0 * (i + 0.5))) : 0.f;
        }
        for (int n = 0; n < 36; ++n)
            for (int k = 0; k < 18; ++k)
                imdct_l[n][k] = static_cast<float>(
                    std::cos(PI / 72.0 * (2 * n + 1 + 18) * (2 * k + 1)));
        for (int n = 0; n < 12; ++n)
            for (int k = 0; k < 6; ++k)
                imdct_s[n][k] = static_cast<float>(
                    std::cos(PI / 24.0 * (2 * n + 1 + 6) * (2 * k + 1)));
        for (int i = 0; i < 64; ++i)
            for (int k = 0; k < 32; ++k)
                synth[i][k] = static_cast<float>(
                    std::cos((16 + i) * (2 * k + 1) * PI / 64.0));
        // The synthesis window is the analysis window times 32 (11172-3,
        // Table 3-B.3 against 3-C.1). The encoder's table is C[i] in Q31.
        for (int i = 0; i < 512; ++i)
            d_win[i] = static_cast<float>(32.0 * tables::enwindow[i] / 2147483648.0);
        for (int sr = 0; sr < 3; ++sr) {
            const auto& sb = tables::scale_fact_band_index_short[kTableIndex[sr]];
            int ti = 0;
            for (int sfb = 0; sfb < tables::NUM_SCFB_SHORT - 1; ++sfb)
                for (int w = 0; w < 3; ++w)
                    for (int f = sb[sfb]; f < sb[sfb + 1]; ++f)
                        short_reorder[sr][ti++] = (f / 6) * 18 + w * 6 + (f % 6);
        }
    }

    void reset() {
        in.clear();
        out.clear();
        resv_len = 0;
        samplerate = 0;
        for (auto& c : overlap)
            for (auto& s : c)
                s.fill(0.f);
        for (auto& c : v)
            c.fill(0.f);
        v_off = {0, 0};
        history_count = 0;
        history_head = 0;
        looping = false;
    }

    std::uint32_t next_rand() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    }

    // ---- Framing ------------------------------------------------------------

    /// Parses a header at `p`. Returns false for anything that is not an
    /// MPEG-1 Layer III header this decoder handles.
    static bool parse_header(const std::uint8_t* p, Frame& f) {
        if (p[0] != 0xFF || (p[1] & 0xE0) != 0xE0)
            return false;
        const int version = (p[1] >> 3) & 3;
        const int layer   = (p[1] >> 1) & 3;
        const int protect = p[1] & 1;
        const int br_idx  = (p[2] >> 4) & 15;
        const int sr_idx  = (p[2] >> 2) & 3;
        const int padding = (p[2] >> 1) & 1;
        if (version != 3 || layer != 1 || br_idx == 0 || br_idx == 15 || sr_idx == 3)
            return false;
        f.samplerate_index = sr_idx;
        f.mode     = (p[3] >> 6) & 3;
        f.mode_ext = (p[3] >> 4) & 3;
        f.channels = (f.mode == 3) ? 1 : 2;
        f.bytes = 144000 * kBitratesV1L3[br_idx] / kSampleratesV1[sr_idx] + padding;
        f.header_bytes = protect ? 4 : 6;
        f.side_bytes = (f.channels == 1) ? 17 : 32;
        return f.bytes > f.header_bytes + f.side_bytes;
    }

    static void parse_side_info(const std::uint8_t* p, Frame& f) {
        BitReader br{p, f.side_bytes * 8, 0};
        const int nch = f.channels;
        f.main_data_begin = static_cast<int>(br.get(9));
        br.get(nch == 1 ? 5 : 3);
        for (int ch = 0; ch < nch; ++ch)
            for (int b = 0; b < 4; ++b)
                f.scfsi[ch][b] = static_cast<int>(br.get(1));
        for (int gr = 0; gr < 2; ++gr) {
            for (int ch = 0; ch < nch; ++ch) {
                Granule& g = f.gr[gr][ch];
                g = Granule{};
                g.part2_3_length    = static_cast<int>(br.get(12));
                g.big_values        = std::min(static_cast<int>(br.get(9)), 288);
                g.global_gain       = static_cast<int>(br.get(8));
                g.scalefac_compress = static_cast<int>(br.get(4));
                g.window_switching  = static_cast<int>(br.get(1));
                if (g.window_switching) {
                    g.block_type = static_cast<int>(br.get(2));
                    g.mixed      = static_cast<int>(br.get(1));
                    for (int r = 0; r < 2; ++r)
                        g.table_select[r] = static_cast<int>(br.get(5));
                    for (int w = 0; w < 3; ++w)
                        g.subblock_gain[w] = static_cast<int>(br.get(3));
                    // A window-switched granule with block_type 0 is illegal;
                    // read it as an ordinary long one.
                } else {
                    for (int r = 0; r < 3; ++r)
                        g.table_select[r] = static_cast<int>(br.get(5));
                    g.region0_count = static_cast<int>(br.get(4));
                    g.region1_count = static_cast<int>(br.get(3));
                }
                g.preflag        = static_cast<int>(br.get(1));
                g.scalefac_scale = static_cast<int>(br.get(1));
                g.count1table    = static_cast<int>(br.get(1));
            }
        }
    }

    /// The rot bend: flips main-data bits at the given rate. The gap to the
    /// next flip is drawn from the geometric distribution, so the cost is per
    /// flip rather than per bit, and the gap carries across frames.
    void rot(std::uint8_t* p, int nbytes) {
        const double rate = bends.rot;
        if (!(rate > 0.0)) {
            rot_skip = -1.0;
            return;
        }
        const double r = std::min(rate, 0.5);
        const double nbits = nbytes * 8.0;
        double pos = 0.0;
        for (;;) {
            if (rot_skip < 0.0) {
                const double u = (next_rand() + 1.0) / 4294967297.0;
                rot_skip = std::floor(std::log(u) / std::log1p(-r));
            }
            if (pos + rot_skip >= nbits) {
                rot_skip -= (nbits - pos);
                return;
            }
            pos += rot_skip;
            const int b = static_cast<int>(pos);
            p[b >> 3] ^= static_cast<std::uint8_t>(0x80u >> (b & 7));
            pos += 1.0;
            rot_skip = -1.0;
        }
    }

    // ---- One granule ----------------------------------------------------------

    void decode_scalefactors(BitReader& br, const Frame& f, int gr, int ch,
                             std::array<int, 22>& sfl,
                             std::array<std::array<int, 13>, 3>& sfs) {
        const Granule& g = f.gr[gr][ch];
        const int slen1 = tables::slen1_tab[g.scalefac_compress];
        const int slen2 = tables::slen2_tab[g.scalefac_compress];
        sfl.fill(0);
        for (auto& w : sfs)
            w.fill(0);
        if (g.window_switching && g.block_type == 2) {
            for (int sfb = 0; sfb < 6; ++sfb)
                for (int w = 0; w < 3; ++w)
                    sfs[w][sfb] = static_cast<int>(br.get(slen1));
            for (int sfb = 6; sfb < 12; ++sfb)
                for (int w = 0; w < 3; ++w)
                    sfs[w][sfb] = static_cast<int>(br.get(slen2));
            return;
        }
        static constexpr int kGroup[5] = {0, 6, 11, 16, 21};
        for (int b = 0; b < 4; ++b) {
            const int len = b < 2 ? slen1 : slen2;
            for (int sfb = kGroup[b]; sfb < kGroup[b + 1]; ++sfb) {
                if (gr == 1 && f.scfsi[ch][b])
                    sfl[sfb] = sf_long_gr0[ch][sfb];
                else
                    sfl[sfb] = static_cast<int>(br.get(len));
            }
        }
        if (gr == 0)
            sf_long_gr0[ch] = sfl;
    }

    void decode_huffman(BitReader& br, int end_bit, const Frame& f, const Granule& g,
                        std::array<int, kLines>& is) {
        const auto& t = trees().t;
        const auto& sfb = tables::scale_fact_band_index[kTableIndex[f.samplerate_index]];
        is.fill(0);

        const int big = std::min(g.big_values * 2, kLines);
        int r1, r2;
        if (g.window_switching) {
            r1 = (g.block_type == 2) ? 36 : sfb[8];
            r2 = kLines;
        } else {
            r1 = sfb[std::min(g.region0_count + 1, 22)];
            r2 = sfb[std::min(g.region0_count + g.region1_count + 2, 22)];
        }

        int i = 0;
        for (; i < big; i += 2) {
            const int region = (i >= r1 ? 1 : 0) + (i >= r2 ? 1 : 0);
            const int ti = g.table_select[region];
            const auto& h = huffman::at(ti);
            if (ti == 0 || !h.code)
                continue;   // all zeros, no bits
            const int code = br.huff(t[ti]);
            int x = code / h.ylen, y = code % h.ylen;
            if (h.linbits && x == 15)
                x += static_cast<int>(br.get(h.linbits));
            if (x && br.bit())
                x = -x;
            if (h.linbits && y == 15)
                y += static_cast<int>(br.get(h.linbits));
            if (y && br.bit())
                y = -y;
            is[i] = x;
            is[i + 1] = y;
        }

        const Tree& quad = t[32 + g.count1table];
        while (i + 4 <= kLines && br.pos < end_bit) {
            const int p = br.huff(quad);
            int q[4] = {p & 1, (p >> 1) & 1, (p >> 2) & 1, (p >> 3) & 1};
            for (int& b : q)
                if (b && br.bit())
                    b = -1;
            if (br.pos > end_bit)
                break;   // ran past the granule: the last quadruple is stuffing
            for (int k = 0; k < 4; ++k)
                is[i + k] = q[k];
            i += 4;
        }
    }

    /// |q|^(4/3), or |q|^law under the law bend.
    float dequant(int q) const {
        const int a = std::min(std::abs(q), 8206);
        const float m = (bends.law == 4.0 / 3.0)
            ? pow43[a]
            : (a ? static_cast<float>(std::pow(static_cast<double>(a), bends.law)) : 0.f);
        return q < 0 ? -m : m;
    }

    /// Requantizes one granule into `xr`, in subband order.
    ///
    /// The law bend is level-neutral: the granule is rescaled so its energy
    /// is what the standard's 4/3 would have given it. A different exponent
    /// then only redistributes the granule's energy among its lines - towards
    /// the loud ones above 4/3, away from them below - and never makes the
    /// whole thing louder or quieter, which an exponent on its own does by
    /// tens of dB.
    void requantize(const Frame& f, const Granule& g, const std::array<int, kLines>& is,
                    const std::array<int, 22>& sfl,
                    const std::array<std::array<int, 13>, 3>& sfs,
                    std::array<float, kLines>& xr) {
        const int sri = kTableIndex[f.samplerate_index];
        const double mult = 0.5 * (1 + g.scalefac_scale);
        const double gain = std::pow(2.0, 0.25 * (g.global_gain - 210));
        const bool bent = bends.law != 4.0 / 3.0;
        double e_std = 0.0, e_bent = 0.0;

        // One line: its bent value, and both energies when bent.
        auto line = [&](int q, double k) -> float {
            if (!q)
                return 0.f;
            const double v = dequant(q) * k;
            if (bent) {
                const double r = pow43[std::min(std::abs(q), 8206)] * k;
                e_std += r * r;
                e_bent += v * v;
            }
            return static_cast<float>(v);
        };

        if (g.window_switching && g.block_type == 2) {
            const auto& sb = tables::scale_fact_band_index_short[sri];
            std::array<float, kLines> coded{};
            int ti = 0;
            for (int s = 0; s < tables::NUM_SCFB_SHORT - 1; ++s) {
                const int width = sb[s + 1] - sb[s];
                for (int w = 0; w < 3; ++w) {
                    const double k = gain * std::pow(2.0, -2.0 * g.subblock_gain[w]
                                                          - mult * sfs[w][s]);
                    for (int j = 0; j < width; ++j, ++ti)
                        coded[ti] = line(is[ti], k);
                }
            }
            const auto& order = short_reorder[f.samplerate_index];
            for (int i = 0; i < kLines; ++i)
                xr[order[i]] = coded[i];
        } else {
            const auto& sfb = tables::scale_fact_band_index[sri];
            for (int s = 0; s < 22; ++s) {
                const int sf = sfl[s] + (g.preflag ? tables::pretab[s] : 0);
                const double k = gain * std::pow(2.0, -mult * sf);
                for (int i = sfb[s]; i < sfb[s + 1]; ++i)
                    xr[i] = line(is[i], k);
            }
        }

        if (bent && e_bent > 0.0) {
            const float norm = static_cast<float>(std::sqrt(e_std / e_bent));
            for (auto& v : xr)
                v *= norm;
        }
    }

    // ---- Filterbank -------------------------------------------------------------

    void antialias(std::array<float, kLines>& xr) const {
        for (int sb = 1; sb < kSubbands; ++sb) {
            for (int i = 0; i < 8; ++i) {
                const int lo = 18 * sb - 1 - i, hi = 18 * sb + i;
                const float a = xr[lo], b = xr[hi];
                xr[lo] = a * cs[i] - b * ca[i];
                xr[hi] = b * cs[i] + a * ca[i];
            }
        }
    }

    /// IMDCT, windowing and overlap for one granule of one channel, then
    /// frequency inversion. Leaves 18 time samples per subband in `t`.
    void hybrid(int ch, int bt, const std::array<float, kLines>& xr,
                std::array<std::array<float, kSubbands>, 18>& t) {
        for (int sb = 0; sb < kSubbands; ++sb) {
            float y[36] = {};
            const float* X = &xr[sb * 18];
            if (bt == 2) {
                for (int w = 0; w < 3; ++w) {
                    for (int n = 0; n < 12; ++n) {
                        float acc = 0.f;
                        for (int k = 0; k < 6; ++k)
                            acc += X[w * 6 + k] * imdct_s[n][k];
                        y[6 + 6 * w + n] += acc * win[2][n];
                    }
                }
            } else {
                const auto& w = win[bt & 3];
                for (int n = 0; n < 36; ++n) {
                    float acc = 0.f;
                    for (int k = 0; k < 18; ++k)
                        acc += X[k] * imdct_l[n][k];
                    y[n] = acc * w[n];
                }
            }
            auto& ov = overlap[ch][sb];
            for (int i = 0; i < 18; ++i) {
                float s = y[i] + ov[i];
                ov[i] = y[i + 18];
                if ((sb & 1) && (i & 1))
                    s = -s;
                t[i][sb] = s;
            }
        }
    }

    /// Polyphase synthesis of one time slot: 32 subband samples in, 32 PCM
    /// samples out.
    void synthesise(int ch, const std::array<float, kSubbands>& s, float* pcm, int stride) {
        int& off = v_off[ch];
        off = (off - 64) & 1023;
        auto& V = v[ch];
        for (int i = 0; i < 64; ++i) {
            float acc = 0.f;
            for (int k = 0; k < 32; ++k)
                acc += synth[i][k] * s[k];
            V[(off + i) & 1023] = acc;
        }
        for (int j = 0; j < 32; ++j) {
            float acc = 0.f;
            for (int i = 0; i < 8; ++i) {
                acc += V[(off + i * 128 + j) & 1023]      * d_win[i * 64 + j];
                acc += V[(off + i * 128 + 96 + j) & 1023] * d_win[i * 64 + 32 + j];
            }
            pcm[j * stride] = acc;
        }
    }

    // ---- One frame --------------------------------------------------------------

    /// A Xing or Info tag rides in an otherwise empty frame at the head of
    /// the stream. It is not audio: decoding it would put a frame of silence
    /// in front of everything, so it is skipped, as every player skips it.
    static bool is_tag_frame(const std::uint8_t* p, const Frame& f) {
        const std::uint8_t* t = p + f.header_bytes + f.side_bytes;
        return f.bytes >= f.header_bytes + f.side_bytes + 4 &&
               (std::memcmp(t, "Xing", 4) == 0 || std::memcmp(t, "Info", 4) == 0);
    }

    void decode_frame(const std::uint8_t* p, Frame& f) {
        if (is_tag_frame(p, f))
            return;
        parse_side_info(p + f.header_bytes, f);
        samplerate = kSampleratesV1[f.samplerate_index];

        // This frame's main data enters the reservoir — bent on the way in —
        // and the frame's own main data starts main_data_begin bytes before it.
        const int md_off = f.header_bytes + f.side_bytes;
        const int md_len = f.bytes - md_off;
        if (resv_len + md_len > kResvBytes) {
            const int drop = resv_len + md_len - kResvBytes;
            std::memmove(resv.data(), resv.data() + drop, resv_len - drop);
            resv_len -= drop;
        }
        const int prev_len = resv_len;
        std::memcpy(resv.data() + resv_len, p + md_off, md_len);
        rot(resv.data() + resv_len, md_len);
        resv_len += md_len;
        const int start = prev_len - f.main_data_begin;

        Spectra spec{};
        const int nch = f.channels;
        if (start >= 0) {
            BitReader br{resv.data(), resv_len * 8, start * 8};
            for (int gr = 0; gr < 2; ++gr) {
                for (int ch = 0; ch < nch; ++ch) {
                    const Granule& g = f.gr[gr][ch];
                    const int gstart = br.pos;
                    const int gend = gstart + g.part2_3_length;
                    std::array<int, 22> sfl;
                    std::array<std::array<int, 13>, 3> sfs;
                    decode_scalefactors(br, f, gr, ch, sfl, sfs);
                    std::array<int, kLines> is;
                    decode_huffman(br, gend, f, g, is);
                    requantize(f, g, is, sfl, sfs, spec.xr[gr][ch]);
                    spec.block_type[gr][ch] =
                        g.window_switching ? g.block_type : 0;
                    br.pos = gend;
                }
                // M/S stereo. Intensity stereo is not produced by the encoder
                // and is read as plain L/R.
                if (nch == 2 && f.mode == 1 && (f.mode_ext & 2)) {
                    constexpr float r = 0.70710678f;
                    auto& m = spec.xr[gr][0];
                    auto& s = spec.xr[gr][1];
                    for (int i = 0; i < kLines; ++i) {
                        const float a = m[i], b = s[i];
                        m[i] = (a + b) * r;
                        s[i] = (a - b) * r;
                    }
                }
                if (nch == 1) {
                    spec.xr[gr][1] = spec.xr[gr][0];
                    spec.block_type[gr][1] = spec.block_type[gr][0];
                }
            }
        }
        // Without its reservoir (the start of a stream, or straight after a
        // reset) the frame decodes as silence, which is what it is.

        // The loop bend. History always records what came in; while looping,
        // what goes out is read back from it instead.
        const Spectra* use = &spec;
        if (bends.loop && history_count > 0) {
            if (!looping) {
                looping = true;
                loop_len = std::clamp(bends.loop_frames, 1, std::min(kMaxLoop, history_count));
                loop_start = (history_head - loop_len + kMaxLoop) % kMaxLoop;
                loop_pos = 0;
            }
            use = &history[(loop_start + loop_pos) % kMaxLoop];
            loop_pos = (loop_pos + 1) % loop_len;
        } else {
            looping = false;
            history[history_head] = spec;
            history_head = (history_head + 1) % kMaxLoop;
            history_count = std::min(history_count + 1, kMaxLoop);
        }

        const std::size_t base = out.size();
        out.resize(base + 1152 * 2);
        for (int gr = 0; gr < 2; ++gr) {
            for (int ch = 0; ch < 2; ++ch) {
                std::array<float, kLines> xr = use->xr[gr][ch];
                const int bt = use->block_type[gr][ch];
                if (bt != 2)
                    antialias(xr);
                std::array<std::array<float, kSubbands>, 18> t;
                hybrid(ch, bt, xr, t);
                float* pcm = out.data() + base + static_cast<std::size_t>(gr) * 576 * 2 + ch;
                for (int slot = 0; slot < 18; ++slot)
                    synthesise(ch, t[slot], pcm + slot * 32 * 2, 2);
            }
        }
    }
};

Decoder::Decoder() : impl_(std::make_unique<Impl>()) {}
Decoder::~Decoder() = default;

DecoderBends& Decoder::bends() noexcept { return impl_->bends; }
int Decoder::samplerate() const noexcept { return impl_->samplerate; }
void Decoder::reset() { impl_->reset(); }

std::span<const float> Decoder::decode(std::span<const std::uint8_t> bytes) {
    auto& d = *impl_;
    d.out.clear();
    d.in.insert(d.in.end(), bytes.begin(), bytes.end());

    std::size_t pos = 0;
    while (pos + 4 <= d.in.size()) {
        Frame f;
        if (!Impl::parse_header(d.in.data() + pos, f)) {
            ++pos;   // hunting for sync
            continue;
        }
        if (pos + static_cast<std::size_t>(f.bytes) > d.in.size())
            break;   // the rest of this frame has not arrived yet
        d.decode_frame(d.in.data() + pos, f);
        pos += static_cast<std::size_t>(f.bytes);
    }
    d.in.erase(d.in.begin(), d.in.begin() + static_cast<std::ptrdiff_t>(pos));
    return {d.out.data(), d.out.size()};
}

} // namespace rmps
