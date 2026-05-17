#include "core/reservoir.hpp"
#include "core/encoder_state.hpp"

namespace rmps::reservoir {

int max_reservoir_bits(double pe_val, EncoderState& s) {
    int mean_bits = s.mean_bits / s.wave.channels;
    int max_bits  = mean_bits;
    if (max_bits > 4095) max_bits = 4095;
    if (!s.ResvMax) return max_bits;

    int more_bits = static_cast<int>(pe_val * 3.1 - mean_bits);
    int add_bits  = 0;
    if (more_bits > 100) {
        int frac = (s.ResvSize * 6) / 10;
        add_bits = (frac < more_bits) ? frac : more_bits;
    }
    int over_bits = s.ResvSize - ((s.ResvMax * 8) / 10) - add_bits;
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

    int over_bits = s.ResvSize - s.ResvMax;
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
