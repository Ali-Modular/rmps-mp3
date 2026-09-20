#pragma once
#include "rmps/rmps.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace rmps::app {

/// One granule of one channel, compact: what the bitstream carries.
struct GranRec {
    std::uint8_t block = 0;          ///< 0 long, 1 start, 2 short, 3 stop
    std::uint8_t scale = 0;
    std::uint8_t preflag = 0;
    std::uint8_t compress = 0;
    std::uint8_t c1tab = 0;
    std::uint8_t r0 = 0, r1 = 0;
    std::array<std::uint8_t, 3> tbl{};
    std::array<std::uint8_t, 3> sbg{};
    std::int16_t gg = 0;
    std::uint16_t part2 = 0, part23 = 0;
    std::uint16_t big_values = 0, count1 = 0;
    std::array<std::int8_t, 21> sf_l{};
    std::array<std::array<std::int8_t, 12>, 3> sf_s{};
};

struct FrameRec {
    std::uint16_t kbps = 0;
    bool ms = false;
    std::int32_t resv = 0, resv_max = 0, overflow = 0;
    float lpf_hz = 0.0f;
    std::uint8_t n_gr = 2, n_ch = 2;
    std::array<std::array<GranRec, 2>, 2> g{};   ///< [gr][ch]
};

/// Everything the analysis view shows for one encode. Written by the encode
/// worker (under `mu`), read by the UI thread.
struct Analysis {
    std::mutex mu;
    std::string title;
    std::string settings;            ///< "ABR 128 kbps, effort 2"
    int target_kbps = 0;             ///< ABR target / CBR rate; 0 for quality mode
    int samplerate = 0;              ///< coded rate
    BandLayout layout;
    std::vector<FrameRec> frames;
    bool finished = false;

    /// Convert and append one frame's stats. Takes the lock.
    void append(const FrameStats& fs);
};

} // namespace rmps::app
