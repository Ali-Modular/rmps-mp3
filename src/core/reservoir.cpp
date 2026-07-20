#include "core/reservoir.hpp"
#include "core/encoder_state.hpp"

#include <algorithm>

namespace rmps::reservoir {

// A large cross-frame reservoir smears one frame's bits into its neighbours
// via main_data_begin. That is fine at higher rates but at low bitrates the
// bank rivals a whole frame, so consecutive frames swing rich→poor and the
// (dead-zone-protected) bass reconstruction pumps frame-to-frame — audible
// as a low "boko-boko" thump. Taper the effective bank to zero across the
// low-rate zone so 64–96 kbps output stays as steady as the pre-reservoir
// encoder, while the full reservoir survives at ≥112 kbps (where the
// pre-echo win was validated). Uses the instantaneous per-frame bitrate so
// ABR — whose per-frame rate swings — tracks it directly.
int effective_resv_max(const EncoderState& s) {
    if (s.ResvMax <= 0 || s.mpeg.bitr <= 0) return s.ResvMax;
    double f = 1.0;
    if (s.mpeg.bitr < 112)
        f = std::clamp((static_cast<double>(s.mpeg.bitr) - 88.0)
                           / (112.0 - 88.0), 0.0, 1.0);
    return static_cast<int>(s.ResvMax * f);
}

int max_reservoir_bits(double pe_val, EncoderState& s) {
    int mean_bits = s.mean_bits / s.wave.channels;
    int max_bits  = mean_bits;
    if (max_bits > 4095) max_bits = 4095;
    const int resv_max = effective_resv_max(s);
    if (!resv_max) return max_bits;

    int more_bits = static_cast<int>(pe_val * 3.1 - mean_bits);
    int add_bits  = 0;
    if (more_bits > 100) {
        int frac = (s.ResvSize * 6) / 10;
        add_bits = (frac < more_bits) ? frac : more_bits;
    }
    int over_bits = s.ResvSize - ((resv_max * 8) / 10) - add_bits;
    if (over_bits > 0) add_bits += over_bits;

    max_bits += add_bits;
    if (max_bits > 4095) max_bits = 4095;
    return max_bits;
}

void adjust(const GrInfo& gi, EncoderState& s) {
    s.ResvSize += (s.mean_bits / s.wave.channels) - gi.part2_3_length;
}

void frame_end(EncoderState& s) {
    auto& l3_side = s.side_info;
    int ancillary_pad = 0;

    if (s.wave.channels == 2 && (s.mean_bits & 1)) {
        s.ResvSize += 1;
    }

    int over_bits = s.ResvSize - effective_resv_max(s);
    if (over_bits < 0) over_bits = 0;
    s.ResvSize -= over_bits;
    int stuffingBits = over_bits + ancillary_pad;

    over_bits = s.ResvSize % 8;
    if (over_bits) {
        stuffingBits += over_bits;
        s.ResvSize   -= over_bits;
    }

    if (!stuffingBits) return;

    // plan a: cram into gr=0/ch=0
    auto& gi0 = l3_side.gr[0].ch[0].tt;
    if (gi0.part2_3_length + stuffingBits < 4095) {
        gi0.part2_3_length += stuffingBits;
        return;
    }

    // plan b: distribute across granules/channels
    for (int gr = 0; gr < s.mpeg.granules_per_frame && stuffingBits > 0; ++gr) {
        for (int ch = 0; ch < s.wave.channels && stuffingBits > 0; ++ch) {
            auto& gi = l3_side.gr[gr].ch[ch].tt;
            int extraBits  = 4095 - gi.part2_3_length;
            int bitsThisGr = (extraBits < stuffingBits) ? extraBits : stuffingBits;
            gi.part2_3_length += bitsThisGr;
            stuffingBits      -= bitsThisGr;
        }
    }
    if (stuffingBits) l3_side.resvDrain = stuffingBits;
}

} // namespace rmps::reservoir
