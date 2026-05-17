#pragma once
#include <array>
#include <cstdint>

namespace rmps::tables {

inline constexpr int NUM_BITRATES = 16; ///< incl. invalid 0 and 15
inline constexpr int NUM_VERSIONS = 4;  ///< MPEG-2.5, reserved, MPEG-II, MPEG-I
inline constexpr int NUM_SAMPLERATES = 9;
inline constexpr int NUM_SCFB_LONG = 23; ///< sfb boundaries (incl. terminator)
inline constexpr int NUM_ENWINDOW = 512;

extern const std::array<std::uint8_t, 16> slen1_tab;
extern const std::array<std::uint8_t, 16> slen2_tab;

extern const std::array<int, NUM_SAMPLERATES> samplerates;

/// bitrates[index][version] in kbps; -1 marks invalid.
extern const std::array<std::array<int, NUM_VERSIONS>, NUM_BITRATES> bitrates;

/// Long-block scalefactor band boundaries, indexed by samplerate_index.
extern const std::array<std::array<int, NUM_SCFB_LONG>, NUM_SAMPLERATES>
    scale_fact_band_index;

/// Subband analysis window (fixed-point, Q?), copied verbatim from Encoder.
extern const std::array<std::int32_t, NUM_ENWINDOW> enwindow;

/// Pretab values for preflag (long blocks, sfb 0..21).
extern const std::array<std::uint8_t, 22> pretab;

/// scfsi band boundaries (long blocks): { 0, 6, 11, 16, 21 }.
extern const std::array<std::uint8_t, 5> scfsi_band_boundaries;

} // namespace rmps::tables
