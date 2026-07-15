#pragma once
#include <array>
#include <cstdint>

namespace rmps {

inline constexpr int MAX_CHANNELS       = 2;
inline constexpr int MAX_GRANULES       = 2;
inline constexpr int GRANULE_SAMPLES    = 576;
inline constexpr int NUM_SFB_LONG       = 21;   // long-block scalefactor bands
inline constexpr int SCALEFAC_L_SIZE    = 22;   // include sentinel
inline constexpr int NUM_SFB_SHORT      = 12;   // short-block scalefactor bands (per window)
inline constexpr int NUM_WINDOWS        = 3;    // short-block windows per granule
inline constexpr int SUBBANDS           = 32;
inline constexpr int K_PER_GRANULE      = 18;
inline constexpr int HAN_SIZE           = 512;

// Block types (window_switching). 0 = normal long, 1 = start (long->short),
// 2 = short (3 windows), 3 = stop/end (short->long).
inline constexpr int BLOCK_LONG  = 0;
inline constexpr int BLOCK_START = 1;
inline constexpr int BLOCK_SHORT = 2;
inline constexpr int BLOCK_STOP  = 3;

struct GrInfo {
    int part2_3_length     = 0;
    int big_values         = 0;
    int count1             = 0;
    int global_gain        = 0;
    int scalefac_compress  = 0;
    std::array<int, 3> table_select {0, 0, 0};
    int region0_count      = 0;
    int region1_count      = 0;
    int preflag            = 0;
    int scalefac_scale     = 0;
    int count1table_select = 0;
    int part2_length       = 0;
    int sfb_lmax           = 0;
    int address1           = 0;
    int address2           = 0;
    int address3           = 0;
    int quantizerStepSize  = 0;
    std::array<int, 4> slen {0, 0, 0, 0};

    // ----- Block-type / window switching (short blocks) -----
    int window_switching_flag = 0;       ///< 1 when block_type != BLOCK_LONG
    int block_type            = BLOCK_LONG;
    int mixed_block_flag      = 0;        ///< v1: always 0 (pure short)
    std::array<int, 3> subblock_gain {0, 0, 0}; ///< per short window (block_type 2)
};

struct ChannelInfo { GrInfo tt; };
struct GranuleInfo { std::array<ChannelInfo, MAX_CHANNELS> ch; };

struct SideInfo {
    int private_bits = 0;
    int resvDrain    = 0;
    std::array<std::array<int, 4>, MAX_CHANNELS> scfsi {};
    std::array<GranuleInfo, MAX_GRANULES> gr;
};

struct Scalefac {
    /// [gr][ch][sfb]
    std::array<
        std::array<
            std::array<int, SCALEFAC_L_SIZE>, MAX_CHANNELS>,
        MAX_GRANULES> l {};

    /// Short-block scalefactors: [gr][ch][window][sfb]. Used when
    /// block_type == BLOCK_SHORT. scfsi/preflag do not apply to short blocks.
    std::array<
        std::array<
            std::array<
                std::array<int, NUM_SFB_SHORT>, NUM_WINDOWS>, MAX_CHANNELS>,
        MAX_GRANULES> s {};
};

} // namespace rmps
