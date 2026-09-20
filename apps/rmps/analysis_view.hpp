#pragma once
#include "analysis.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace rmps::app {

/// ImGui panel that visualises an `Analysis`: bitrate decisions, reservoir,
/// block types, scalefactors, Huffman table selection and per-frame detail.
///
/// Keeps its own copy of the frames (only the new ones are copied each UI
/// frame), so drawing never holds the encoder's lock.
class AnalysisView {
public:
    AnalysisView();
    ~AnalysisView();
    void draw(Analysis* a);

private:
    struct Tex;
    void sync(Analysis* a);
    void tab_overview();
    void tab_scalefactors();
    void tab_huffman();
    void tab_frame();
    void tab_stats();

    // --- mirrored data ---
    Analysis* src_ = nullptr;
    std::vector<FrameRec> fr_;
    std::string title_, settings_;
    int target_kbps_ = 0;
    int samplerate_ = 44100;
    BandLayout layout_;
    bool finished_ = false;
    std::array<std::array<double, 35>, 2> huff_lines_{};  ///< [ch][category] line counts
    std::vector<int> short_tx_of_freq_;                   ///< long-equivalent line -> short tx index

    // --- view state ---
    double v0_ = 0.0, v1_ = 1.0;   ///< visible frame range
    bool follow_ = true;            ///< track growth until the user zooms
    int sel_ = 0;                   ///< selected frame
    int channel_ = 0;
    int sf_mode_ = 0;               ///< 0 scalefactor value, 1 effective step dB
    std::unique_ptr<Tex> sf_tex_, huff_tex_;
};

} // namespace rmps::app
