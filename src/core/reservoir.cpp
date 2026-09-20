#include "core/reservoir.hpp"
#include <cstdio>
#include "core/encoder_state.hpp"

#include <algorithm>
#include <cstdlib>

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
// ...but tapering it to *zero* throws out the baby with the bathwater, and
// that is what the floor below fixes. At 64 kbps the taper made
// `max_reservoir_bits` return before it did anything at all, so no granule
// could ever draw and — worse — `frame_end` clamps `ResvSize` to this value,
// so every surplus bit a quiet granule left behind was emitted as 0xFF
// stuffing instead of being banked. Measured on a 232 s program at 64 kbps:
// bank capacity 0, and therefore an attack granule with nothing to borrow.
// That is the same shortfall that shows up as pre-echo down there.
//
// A floor of 30 % keeps a bank small enough that it cannot smear a whole
// frame into its neighbour (the pumping this taper was written for needs the
// bank to rival a frame) while still funding the one thing low rates most
// need it for: the attack draw in `iteration_loop`. The general, pe-driven
// draw stays disabled in the tapered zone — see `max_reservoir_bits` — so the
// bank there fills from surplus and empties only into attacks.
constexpr double LOW_RATE_RESV_FLOOR = 0.30;

/// True in the low-rate zone where the bank exists but only attacks may spend
/// it. `max_reservoir_bits` uses this to suppress the ordinary demand draw.
bool low_rate_zone(const EncoderState& s) {
    return s.policy_bitr() > 0 && s.policy_bitr() < 112;
}

int effective_resv_max(const EncoderState& s) {
    if (s.ResvMax <= 0 || s.mpeg.bitr <= 0) return s.ResvMax;
    static const double floor_pct = [] {
        if (const char *e = std::getenv("RMPS_RESV_FLOOR"))
            return std::atof(e);
        return LOW_RATE_RESV_FLOOR;
    }();
    double f = 1.0;
    if (s.policy_bitr() < 112)
        f = std::clamp((static_cast<double>(s.policy_bitr()) - 88.0)
                           / (112.0 - 88.0),
                       floor_pct, 1.0);
    return static_cast<int>(s.ResvMax * f);
}

/// Largest share of its flat grant a low-demand granule may hand back to the
/// bank, in percent. This is the knob that decides how hard the reservoir
/// circulates: 0 reproduces the old behaviour (flat grants, idle bank), and
/// too high starves ordinary granules on the strength of a `pe` estimate that
/// was never meant to be exact.
constexpr int GIVEBACK_MAX_PCT = 20;

/// How much of a frame's flat share is held back when the lookahead says an
/// attack lands in the next frame.
///
/// **Currently 0: the mechanism is wired and verified, the policy is not.**
/// The obvious first policy — hold 15 % ahead of a detected attack — measured
/// clearly negative (segSNR 2 cells better / 9 worse, p05 3 / 10). The paying
/// side is reliable and the collecting side is not: every frame before an
/// attack gives up its 15 %, but the attack frame can only take it back
/// through `resv_draw_per_slot`, which draws 70 % of the bank split across
/// attack slots and is then clamped by both the 4095-bit part2_3 field and
/// `frame_hard_bits`. A hold that is certain to be paid and uncertain to be
/// spent is a loss however small it is.
///
/// What to try before raising this again, in order:
///   1. Measure how often `probe_demand` actually fires. An 8× sub-window
///      step is the transient gate's order of magnitude, but the probe is
///      stateless and cruder; if it fires on most frames of ordinary music,
///      the hold is a flat tax and no size of it will pay.
///   2. Make the draw side able to spend the bank — the caps above are what
///      strand it, not the deposit rate.
///   3. Only then size the hold, and size it from the *measured* shortfall of
///      attack granules rather than as a fraction of the flat share.
constexpr int LOOKAHEAD_HOLD_PCT = 0;

int max_reservoir_bits(double pe_val, EncoderState& s) {
    int mean_bits = s.mean_bits / s.wave.channels;
    int max_bits  = mean_bits;
    if (max_bits > 4095) max_bits = 4095;
    const int resv_max = effective_resv_max(s);
    if (!resv_max) return max_bits;

    int more_bits = static_cast<int>(pe_val * 3.1 - mean_bits);
    int add_bits  = 0;
    // In the low-rate zone the bank is deliberately small and reserved for
    // attacks (see LOW_RATE_RESV_FLOOR): letting every busy granule draw on it
    // is the frame-to-frame swing that made the bass pump. Deposits still
    // happen, so the bank is there when `resv_draw_per_slot` needs it.
    if (more_bits > 100 && !low_rate_zone(s)) {
        int frac = (s.ResvSize * 6) / 10;
        add_bits = (frac < more_bits) ? frac : more_bits;
    } else if (more_bits < -100) {
        // Deposit side. Without it the bank has no income: a granule that
        // needs less than its flat share still gets the whole share, spends
        // it (the search fills whatever budget it is given), and the next
        // attack finds an empty bank — which is exactly what the meter showed
        // before `pe` was live (occupancy 0-9 %, `used` within 2 bits of
        // `mean` at every bitrate).
        //
        // Hand back what the demand estimate says is surplus, bounded: `pe` is
        // an estimate, and a granule that is quiet by the model but not by the
        // ear should lose some precision, never fall off a cliff. The cap is
        // on the give-back, not on the resulting grant, so the floor is a
        // fixed fraction of the flat share.
        const int surplus  = -more_bits - 100;
        const int give_cap = (mean_bits * GIVEBACK_MAX_PCT) / 100;
        max_bits -= (surplus < give_cap) ? surplus : give_cap;
    }

    // Lookahead: an attack lands in the *next* frame. Bank for it now.
    //
    // This is the half the demand signal cannot supply on its own. `pe`
    // describes the granule being coded, so a draw only ever happens once the
    // expensive frame is already here — and by then the bank holds whatever
    // the preceding frames happened to leave, which is the arrangement that
    // measured 0-9 % occupancy. Giving up a slice of an ordinary frame to a
    // known-imminent attack is the trade the reservoir exists to make, and it
    // is the only one that needs the future to be visible.
    if (s.lookahead_valid && s.lookahead_attack) {
        const int hold = (mean_bits * LOOKAHEAD_HOLD_PCT) / 100;
        max_bits -= hold;
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
    s.frame_overflow_bits = over_bits;
    int stuffingBits = over_bits + ancillary_pad;

    // An overdrawn reservoir (ResvSize < 0) used to turn the byte-alignment
    // remainder below into *negative* stuffing: C++ `%` keeps the dividend's
    // sign, so -7 % 8 is -7, and "stuffing" -7 bits shortened gr0/ch0's
    // declared part2_3_length below the Huffman data already counted into it.
    // The decoder then reads that granule short and desynchronises -
    // libmpg123 reports "dequantization failed" for the rest of the
    // reservoir. Found on a 128 kbps drum-stem encode: frame 141 went
    // 2757 -> 2750 declared bits with ix, big_values and count1 unchanged,
    // exactly the 7-bit overrun the writer saw. Take the remainder in [0, 8)
    // so stuffing can only ever lengthen a granule.
    if (s.ResvSize < 0 && std::getenv("RMPS_RESV_NEG"))
      std::fprintf(stderr, "[resv] overdrawn by %d bits at frame end\n",
                   -s.ResvSize);
    over_bits = s.ResvSize % 8;
    if (over_bits < 0)
      over_bits += 8;
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
