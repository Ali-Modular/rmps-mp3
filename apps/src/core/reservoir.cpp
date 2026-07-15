#include "core/reservoir.hpp"
#include "core/encoder_state.hpp"

#include <cstdint>

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
    } else if (more_bits < -100 && s.ResvSize < (s.ResvMax * 9) / 10) {
        // Donation: the beam search always spends its whole grant, so the
        // reservoir never fills by itself — easy granules must be given a
        // slightly smaller grant on purpose. Cap the donation at 30% of the
        // share, and stop donating once the reservoir is nearly full (the
        // surplus would only turn into stuffing).
        int donate = -more_bits;
        const int cap = (mean_bits * 3) / 10;
        if (donate > cap) donate = cap;
        add_bits = -donate;
    }
    int over_bits = s.ResvSize - ((s.ResvMax * 8) / 10) - add_bits;
    if (over_bits > 0) add_bits += over_bits;

    max_bits += add_bits;
    if (max_bits > 4095) max_bits = 4095;
    if (max_bits < 0) max_bits = 0;
    return max_bits;
}

void adjust(const GrInfo& gi, EncoderState& s) {
    s.ResvSize += (s.mean_bits / s.wave.channels) - gi.part2_3_length;
}

void frame_end(EncoderState& s) {
    auto& l3_side = s.side_info;

    // Exact frame accounting. The incremental adjust() view sums truncated
    // per-granule shares and drifts by a few bits per frame; main_data_begin
    // needs byte-exact numbers, so recompute the reservoir directly from
    // what the frame provides vs what the granules actually used.
    const int available = s.mpeg.bits_per_frame - s.sideinfo_len;
    int used = 0;
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr)
        for (int ch = 0; ch < s.wave.channels; ++ch)
            used += l3_side.gr[gr].ch[ch].tt.part2_3_length;

    std::int64_t resv =
        static_cast<std::int64_t>(s.resv_frame_start) + available - used;
    int stuffing = 0;
    if (resv < 0) resv = 0;  // unreachable: iteration_loop caps the budgets
    if (resv > s.ResvMax) {
        stuffing = static_cast<int>(resv - s.ResvMax);
        resv = s.ResvMax;
    }
    // Keep the reservoir byte-aligned: each frame's total main_data must end
    // on a byte boundary for the byte-valued main_data_begin to stay exact.
    const int align = static_cast<int>(resv & 7);
    stuffing += align;
    resv -= align;
    s.ResvSize = static_cast<int>(resv);

    // Surplus the reservoir cannot hold is written as ancillary bits after
    // the last granule's main data (format_bitstream consumes resvDrain).
    l3_side.resvDrain = stuffing;
}

} // namespace rmps::reservoir
