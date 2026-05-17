#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace rmps {

/// Per-granule encode stats.
struct GranuleStats {
    int gr            = 0;
    int ch            = 0;
    int gain          = 0;
    int preflag       = 0;
    int scale         = 0;
    int compress      = 0;
    std::array<int, 4> scfsi_pat {0, 0, 0, 0};
    int bits          = 0;     ///< part2 + part3 used
    int budget        = 0;
    double raw_sdr_db = 0.0;
    double perc_sdr_db= 0.0;
    int hf_rdo_bits_saved = 0;
};

/// HFAdaptor visible state.
struct HFAdaptorState {
    double atten_db   = 0.0;
    bool   transient  = false;
    double shelf_hz   = 6000.0;
};

/// PID/ABR controller state (only meaningful when ABR is active).
struct AbrState {
    double target_kbps      = 0.0;
    double avg_kbps         = 0.0;
    double decision_kbps    = 0.0;
    double err              = 0.0;
    double u                = 0.0;
    double integral         = 0.0;
    double min_kbps         = 0.0;
    double max_kbps         = 0.0;
    long   frames           = 0;
};

/// Aggregated per-frame stats; emitted after every frame.
struct FrameStats {
    std::vector<GranuleStats> granules;
    int    total_bits        = 0;
    int    total_budget      = 0;
    double mean_raw_sdr_db   = 0.0;
    double mean_perc_sdr_db  = 0.0;
    int    hf_rdo_bits_saved = 0;
    HFAdaptorState hf {};
    double lpf_side_hz       = 0.0;
};

} // namespace rmps
