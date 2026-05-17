#pragma once
#include <cstdint>
#include <optional>

namespace rmps {

enum class Mode : std::uint8_t {
    Stereo    = 0,
    JointStereo = 1,
    DualChannel = 2,
    Mono      = 3,
};

enum class Emphasis : std::uint8_t {
    None   = 0,
    Mu5015 = 1,
    CITT   = 3,
};

enum class MpegVersion : std::uint8_t {
    Mpeg25 = 0,
    MpegII = 2,
    MpegI  = 3,
};

/// Public encoder configuration.
struct Config {
    int channels    = 2;                 ///< 1 (mono) or 2 (stereo)
    int samplerate  = 44100;             ///< Hz; valid: 8/11.025/12/16/22.05/24/32/44.1/48k
    int bitrate     = 128;               ///< kbps (CBR default, ABR initial)
    Mode mode       = Mode::JointStereo; ///< forced to JointStereo for MS when stereo
    Emphasis emph   = Emphasis::None;
    bool copyright  = false;
    bool original   = true;

    // --- Adaptive features ---
    /// Optional ABR target in kbps. nullopt => CBR at `bitrate`.
    std::optional<int> abr_target_kbps {};
    int abr_min_kbps = 0;                ///< 0 = use abr_target / 2
    int abr_max_kbps = 0;                ///< 0 = use abr_target * 8/5
    /// Optional perceptual SDR floor (dB) for ABR; nullopt disables steering.
    std::optional<double> target_sdr_db {};

    // --- Quality knobs ---
    int  beam_width            = 8;
    int  gain_window           = 1;      ///< ±N around bin-search seed
    int  sf_max_candidate      = 3;
    double noise_shaping_alpha = 0.5;    ///< 0..0.95
    double hf_max_atten_m_db   = 1.0;    ///< HFAdaptor cap (M)
    double hf_max_atten_s_db   = 1.0;    ///< HFAdaptor cap (S)
};

} // namespace rmps
