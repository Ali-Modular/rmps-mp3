#include "control/abr_pid.hpp"
#include "core/encoder_state.hpp"
#include "core/tables.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace rmps::control {

namespace {

/// One extra bit per sample buys about 6 dB of SNR, and one extra bit per
/// sample is roughly a doubling of the rate. Working the quality loop
/// multiplicatively against this keeps its response even across the whole
/// 32-320 kbps span, which a linear gain cannot do.
constexpr double DB_PER_DOUBLING = 6.0;

/// How much better an SDR a quiet frame is asked for, per dB it sits below
/// the reference level. 0 = constant SDR (what the loop did); 1 = constant
/// absolute noise floor. Calibrated against the tilt a rate-matched CBR
/// encode produces, which is the behaviour listeners prefer.
/// Sweepable with RMPS_QUIET_TILT.
const double QUIET_TILT = [] {
    if (const char* e = std::getenv("RMPS_QUIET_TILT")) {
        const double v = std::atof(e);
        if (v >= 0.0 && v <= 1.0) return v;
    }
    return 0.30;
}();

/// Compression exponent on the feed-forward demand (see `set_demand`).
///
/// Left at 1.0 — no compression. It was added to bend the demand's dynamic
/// range and then measured to have nothing to bend: `pe` spans only 1.7 dB
/// from the 5th to the 95th percentile of real frames, so every value of this
/// exponent produces effectively the same allocation. Kept as a hook because
/// the mapping is the right place to shape demand if `pe` ever gains range.
/// Sweepable with RMPS_PE_GAMMA.
const double PE_DEMAND_GAMMA = [] {
    if (const char* e = std::getenv("RMPS_PE_GAMMA")) {
        const double v = std::atof(e);
        if (v >= 0.0 && v <= 1.0) return v;
    }
    return 1.0;
}();

int find_nearest_bitrate_index(double kbps, int version, int& snapped) {
    int best_idx = 1;
    int best_diff = INT_MAX;
    int best_kbps = 0;
    for (int i = 1; i < tables::NUM_BITRATES - 1; ++i) {
        const int b = tables::bitrates[i][version];
        if (b < 0) continue;
        const int diff = std::abs(static_cast<int>(std::lround(kbps)) - b);
        if (diff < best_diff) {
            best_diff = diff;
            best_idx  = i;
            best_kbps = b;
        }
    }
    snapped = best_kbps;
    return best_idx;
}

} // namespace

int switch_bitrate(EncoderState& s, double new_kbps) {
    int snapped = 0;
    const int idx = find_nearest_bitrate_index(new_kbps, s.mpeg.version, snapped);
    if (idx == s.mpeg.bitrate_index && snapped == s.mpeg.bitr) return snapped;

    s.mpeg.bitrate_index = idx;
    s.mpeg.bitr          = snapped;
    set_frame_slots(s.mpeg, s.wave.samplerate);
    return snapped;
}

AbrController::AbrController(Options opt) : o_(std::move(opt)) {
    const int version = o_.mpeg_version;
    int lo = INT_MAX, hi = 0;
    for (int i = 1; i < tables::NUM_BITRATES - 1; ++i) {
        const int b = tables::bitrates[i][version];
        if (b < 0) continue;
        lo = std::min(lo, b);
        hi = std::max(hi, b);
    }
    if (o_.min_kbps <= 0.0) {
        const double target_min = o_.target_kbps - 12.0;
        int best_b = hi;
        for (int i = 1; i < tables::NUM_BITRATES - 1; ++i) {
            const int b = tables::bitrates[i][version];
            if (b < 0) continue;
            if (b >= target_min && b < best_b) {
                best_b = b;
            }
        }
        o_.min_kbps = static_cast<double>(best_b);
    }
    if (o_.max_kbps <= 0.0) {
        const double target_max = o_.target_kbps + 48.0;
        int best_b = hi;
        for (int i = 1; i < tables::NUM_BITRATES - 1; ++i) {
            const int b = tables::bitrates[i][version];
            if (b < 0) continue;
            if (b >= target_max && b < best_b) {
                best_b = b;
                break;
            }
        }
        o_.max_kbps = static_cast<double>(best_b);
    }
    if (o_.quality_mode) {
        // Quality mode has no rate to defend, so the controller gets the whole
        // legal range to work in. Anything narrower would turn a quality target
        // into a rate target by the back door.
        o_.min_kbps = static_cast<double>(lo);
        o_.max_kbps = static_cast<double>(hi);
    }
    legal_min_kbps_ = static_cast<double>(lo);
    o_.min_kbps = std::max(o_.min_kbps, static_cast<double>(lo));
    o_.max_kbps = std::min(o_.max_kbps, static_cast<double>(hi));
    last_decision_kbps_ = o_.target_kbps;
    last_avg_kbps_      = o_.target_kbps;
}

void AbrController::set_demand(double pe_bits, double frame_seconds) noexcept {
    if (!o_.quality_mode || frame_seconds <= 0.0 || pe_bits <= 0.0) return;
    // `pe` counts the bits needed to push each band's quantization noise under
    // its mask. Real coded frames need more than that (Huffman inefficiency,
    // scalefactors, side info) and simultaneously less (nobody codes every
    // band to NMR = 0), so the units do not line up on their own. The gain
    // below is only a starting point: the trim integrates the residual away,
    // so its exact value sets the length of the startup transient, not the
    // steady-state rate.
    constexpr double PE_TO_KBPS = 1.0;
    double ff_kbps = PE_TO_KBPS * pe_bits / frame_seconds / 1000.0;

    // Compress the demand's dynamic range around a reference.
    //
    // `pe` answers "how many bits to put this frame's noise under its mask",
    // and following it exactly is the textbook allocation. Measured against a
    // rate-matched CBR on quiet material, it moves 2.9 dB of error/signal out
    // of the quietest fifth of frames and into the loudest — the model's own
    // optimum, and audibly the wrong trade. Masking models are calibrated on
    // steady tones and noise; they over-credit masking exactly where the
    // material is sparse and tonal, which is what a decay tail or a held note
    // in a quiet passage is, and where the noise is least covered.
    //
    // So follow the *shape* of the demand but not its full depth. gamma = 1
    // is the raw pe allocation, gamma = 0 is flat (CBR-like); in between, a
    // frame reading half the reference demand gets 2^-gamma of the rate
    // instead of half. The reference is the observed median demand, so the
    // compression is centred and the mean rate barely moves — the trim
    // absorbs whatever is left.
    constexpr double PE_REF_KBPS = 100.0;
    if (PE_DEMAND_GAMMA < 1.0 && ff_kbps > 0.0)
        ff_kbps = PE_REF_KBPS * std::pow(ff_kbps / PE_REF_KBPS,
                                         PE_DEMAND_GAMMA);
    last_ff_kbps_ = ff_kbps;
    last_decision_kbps_ =
        std::clamp(ff_kbps * std::pow(2.0, trim_db_ / DB_PER_DOUBLING),
                   o_.min_kbps, o_.max_kbps);
    // Same overflow ceiling as ABR (see `observe`): a quality target never
    // needs bits the reservoir is already throwing away.
    pre_ceiling_decision_kbps_ = last_decision_kbps_;
    ovf_binding_ = ovf_ceiling_kbps_ > 0.0 && last_decision_kbps_ > ovf_ceiling_kbps_;
    if (ovf_binding_)
        last_decision_kbps_ = std::max(ovf_ceiling_kbps_, o_.min_kbps);
    if (std::getenv("RMPS_QVBR_DEBUG"))
        std::fprintf(stderr, "[ff] pe=%.0f ff=%.1f trim=%+.2f -> %.1f kbps\n",
                     pe_bits, ff_kbps, trim_db_, last_decision_kbps_);
}

void AbrController::observe(int frame_bits, double frame_seconds,
                            std::optional<double> frame_sdr_db,
                            std::optional<double> frame_level_db,
                            int overflow_bits) {
    if (frame_seconds <= 0.0) return;

    // --- Reservoir overflow feedback (default OFF) ----------------------
    //
    // When the bank is already full, whatever a frame does not spend is thrown
    // away as stuffing, and the controller never saw that: it is fed the
    // produced frame size, which counts stuffing as spent.
    //
    // Measured first (RMPS_OVF_DEBUG=1): on the four real-music excerpts the
    // bank **never overflowed** — 0 frames in every mode (-b 48/128, --abr
    // 64/128, -q 20/50). The quantizer spends whatever grant it gets. Overflow
    // is a sparse-material effect: at --abr 128, attack_from_silence is 73 %
    // stuffing, transient_train 76 %, sweep 52 %, harmonic_tone 8.6 %.
    //
    // So on music this code does nothing either way. On sparse material every
    // variant tried traded something away (testset, --abr 64/128 and -q 50):
    //
    //  * Averaging useful bits (produced - overflow): harmonic_tone waste
    //    8.55 -> 0.61 %, but the average then reads under target and the P
    //    term pushes the rate up — two_tone 137.6 -> 144 kbps, more waste.
    //    (RMPS_OVF_USEFUL=1.)
    //  * A ceiling learned from overflow, clamped to the ABR band floor:
    //    inert, because the band is [target-12, target+48] and silence at
    //    128 cannot go under 116.
    //  * The same ceiling down to the legal minimum (current code): sweep
    //    132.8 -> 60.1 kbps with waste 52 -> 17 %, but segSNR -1.9 dB, and
    //    transient_train holes 21.5 -> 27.7 % at 64 / 6.1 -> 8.5 % at 128.
    //  * The ceiling in quality mode: after a silent run it sits at the floor
    //    and starves the attack — transient_train holes 4.75 -> 14.75 % even
    //    with an immediate release on transient hints. (RMPS_OVF_QUALITY=1.)
    //
    // Overflow is a lagging signal and an attack after silence needs its bits
    // in the same frame, so a per-frame rate cut learned from overflow keeps
    // meeting the attack too late. Kept for measurement; RMPS_OVF_FEEDBACK=1
    // enables the ABR ceiling.
    static const bool ovf_on = [] {
        const char* e = std::getenv("RMPS_OVF_FEEDBACK");
        return e && std::atoi(e) != 0;
    }();
    if (!ovf_on) overflow_bits = 0;
    const int useful_bits = std::max(0, frame_bits - std::max(0, overflow_bits));
    {
        /// Headroom over the demonstrated useful rate. The reservoir keeps
        /// working under the ceiling, so it does not need to be generous.
        constexpr double OVF_MARGIN = 1.15;
        /// Relaxation per non-overflowing frame: ~+19 % per 0.1 s at 44.1 kHz.
        constexpr double OVF_RELAX = 1.05;
        if (overflow_bits > 0) {
            const double useful_kbps =
                useful_bits / frame_seconds / 1000.0 * OVF_MARGIN;
            const double c = std::max(useful_kbps, legal_min_kbps_);
            ovf_ceiling_kbps_ = (ovf_ceiling_kbps_ > 0.0)
                                    ? std::min(ovf_ceiling_kbps_, c)
                                    : c;
            // Never hold it under the demonstrated use of the frame that
            // just overflowed: a sharper drop than that is a guess.
            ovf_ceiling_kbps_ = std::max(ovf_ceiling_kbps_, c * 0.85);
        } else if (ovf_ceiling_kbps_ > 0.0) {
            ovf_ceiling_kbps_ *= OVF_RELAX;
            if (ovf_ceiling_kbps_ >= o_.max_kbps) ovf_ceiling_kbps_ = 0.0;
        }
    }
    // Diagnostic switches for the two halves (see the measurements in the
    // header comment): RMPS_OVF_USEFUL=1 averages useful bits instead of
    // produced ones; RMPS_OVF_QUALITY=0 keeps the ceiling out of quality mode.
    static const bool use_useful = [] {
        const char* e = std::getenv("RMPS_OVF_USEFUL");
        return e && std::atoi(e) != 0;
    }();
    static const bool ovf_quality = [] {
        const char* e = std::getenv("RMPS_OVF_QUALITY");
        return e && std::atoi(e) != 0;
    }();
    if (o_.quality_mode && !ovf_quality) ovf_ceiling_kbps_ = 0.0;
    if (use_useful) frame_bits = useful_bits;
    // Warm-start the running average with a virtual history at the target, so
    // the cumulative mean does not wind up from the first (often max-bitrate
    // cold-start) frame and drag the PID through a large convergence ramp.
    // That startup ramp is the dominant source of HF-cutoff "swirl" under ABR
    // (the steady-state cutoff jitter already matches CBR). ~25 frames ≈ 0.65 s.
    if (frames_ == 0) {
        constexpr double WARM_FRAMES = 25.0;
        cum_seconds_ = WARM_FRAMES * frame_seconds;
        cum_bits_    = WARM_FRAMES * o_.target_kbps * 1000.0 * frame_seconds;
        cum_seconds_local_ = WARM_FRAMES * frame_seconds;
        cum_bits_local_    = WARM_FRAMES * o_.target_kbps * 1000.0 * frame_seconds;
    }
    cum_bits_    += frame_bits;
    cum_seconds_ += frame_seconds;
    
    // Update local average with a decay factor (~20 frames window)
    constexpr double LOCAL_DECAY = 0.95;
    cum_bits_local_    = LOCAL_DECAY * cum_bits_local_ + frame_bits;
    cum_seconds_local_ = LOCAL_DECAY * cum_seconds_local_ + frame_seconds;
    
    ++frames_;

    const double avg_local = (cum_bits_local_ / cum_seconds_local_) / 1000.0;
    const double avg_global = (cum_bits_ / cum_seconds_) / 1000.0;
    last_avg_kbps_ = avg_global;

    if (o_.quality_mode) {
        // --- Quality-targeted VBR -------------------------------------------
        //
        // Feed-forward from perceptual entropy, trimmed by feedback on SDR.
        // `set_demand` supplies the per-frame shape; this supplies the level.
        if (!frame_sdr_db) return;   // nothing measured; hold the last trim

        // The loop no longer chooses the rate — `set_demand` does, from the
        // frame's own perceptual entropy. All that is left here is the slow
        // trim that pulls the long-run SDR onto the `-q` setpoint.
        //
        // This is also why there is no quiet-passage penalty any more. Earlier
        // versions asked quiet frames for a better SDR than loud ones, first
        // against a decaying running maximum (history-dependent: the same
        // passage scored differently depending on what preceded it) and then
        // against a fixed -14 dBFS anchor (arbitrary, and it fought the `-q`
        // scale for control of the rate). Both were treating a symptom in the
        // wrong layer: quiet frames were not weak because SDR is a ratio, they
        // were weak because a feedback loop cannot follow a decay. At
        // 320 kbps, where the loop never has to react, quiet frames measured
        // 2.9 dB *better* than loud ones — there was never a model defect to
        // compensate for.
        //
        // What remains is that a constant SDR setpoint is itself the wrong
        // goal, and the loop hits it exactly: measured per loudness quintile
        // it holds -23.4 to -24.0 dB err/sig from the quietest fifth of
        // frames to the loudest, dead flat. A rate-matched CBR encode of the
        // same material does not — it runs 3.8 dB *better* in the quiet fifth
        // and worse in the loud one — and that is the version listeners
        // prefer. Constant relative quality means the noise floor rides the
        // signal down, which would be right if the listener rode the volume
        // knob with it. They set it once, and masking weakens as the level
        // drops, so a quiet passage needs a better ratio, not the same one.
        //
        // So tilt the setpoint with the frame level. Only the *slope* is a
        // real parameter: the reference is arbitrary, because shifting it
        // only adds a constant to the setpoint and the integrator removes any
        // constant. That is what the two earlier attempts got wrong — both
        // clamped the term (max(0,...) into a saturating range), which made
        // it nonlinear and therefore anchor-sensitive, and then the anchor
        // had to be guessed. A plain linear tilt has nothing to guess.
        double target = *o_.target_sdr_db;
        if (frame_level_db) {
            constexpr double LEVEL_REF_DBFS = -14.0;  // arbitrary; see above
            target += QUIET_TILT * (LEVEL_REF_DBFS - *frame_level_db);
        }
        const double err = target - *frame_sdr_db;
        // Conditional integration (anti-windup).
        //
        // Without it the integrator kept accumulating while the rate was
        // already pinned at a clamp, and there is nothing at the other end to
        // unwind it: measured at `-q 20`, `trim_db_` reached -64 dB within a
        // hundred frames — 10 doublings past the legal range — so when a frame
        // finally did want more bits (err +1.25 dB) the trim was still -63 and
        // the loop could not respond for hundreds of frames. A setpoint the
        // encoder cannot reach even at `min_kbps` is the normal case at the
        // quiet end of `-q`, not an exceptional one, so this is the difference
        // between a loop that rests at its clamp and one that is destroyed by
        // reaching it.
        //
        // Integrate only when the demanded correction is not already saturated
        // in the direction the error would push it, and hard-bound the state to
        // the range that can actually span the legal bitrates.
        const double span_db =
            DB_PER_DOUBLING * std::log2(std::max(o_.max_kbps, 1.0) /
                                        std::max(o_.min_kbps, 1.0));
        const double trim_lo = -span_db, trim_hi = span_db;
        const bool pinned_low  = trim_db_ <= trim_lo + 1e-9 && err < 0.0;
        const bool pinned_high = trim_db_ >= trim_hi - 1e-9 && err > 0.0;
        if (!is_silent_frame_ && !pinned_low && !pinned_high)
            integral_ = std::max(0.995, o_.integral_leak) * integral_ + err;
        else
            integral_ = std::max(0.995, o_.integral_leak) * integral_;
        // Bound the integrator itself too: the leak alone only bounds it in
        // proportion to the error, and the errors here are tens of dB.
        if (o_.ki > 0.0)
            integral_ = std::clamp(integral_, trim_lo / o_.ki, trim_hi / o_.ki);
        const double deriv = err - prev_err_;
        prev_err_ = err;
        last_err_ = err;

        const double u = o_.kp * err + o_.ki * integral_ + o_.kd * deriv;
        last_u_ = u;

        // Smooth the *trim*, not the decision. Smoothing the decision was what
        // made the old loop sluggish: it low-passed the feed-forward path
        // along with the feedback one. Here the demand reaches the rate
        // untouched and only the correction is slewed.
        //
        // No special case for "silent". That flag comes from the transient
        // gate, where it means "quiet enough that an attack after it counts as
        // from-silence" — it fires at about -63 dBFS, which is a decay tail,
        // not an empty frame. Using it as a bit-allocation instruction slammed
        // exactly those tails to the floor, and a reverb tail is noise-like and
        // exposed: the worst place to spend nothing.
        trim_db_ = std::clamp(o_.smoothing * u + (1.0 - o_.smoothing) * trim_db_,
                              trim_lo, trim_hi);
        if (std::getenv("RMPS_QVBR_DEBUG"))
            std::fprintf(stderr,
                         "[qvbr] sdr=%.2f err=%+.2f u=%+.3f trim=%+.2f "
                         "silent=%d\n",
                         *frame_sdr_db, err, u, trim_db_,
                         static_cast<int>(is_silent_frame_));
        return;
    }

    // Local error for Proportional and Derivative terms (incorporating quality/SDR target)
    double err_local = o_.target_kbps - avg_local;
    if (o_.target_sdr_db && frame_sdr_db) {
        // Bound the quality-floor term.
        //
        // It is written as a floor — "spend more if the measured quality is
        // under target" — but the default target is 90 dB and the metric's
        // working range is roughly 20-60 dB, so the term never goes to zero:
        // at `sdr_weight` 0.7 it contributes a permanent +31 kbps to a
        // 128 kbps target. That is not a floor, it is an offset, and the only
        // thing removing it is the integrator slowly winding against it —
        // which is why ABR settled 8-10 % above target on 30 s of real
        // material and 22 % above on short clips that never got that far.
        //
        // Bounding it at a tenth of the target keeps a genuine shortfall
        // acting on the rate while capping what an unreachable setpoint can
        // bias the average by. The right long-term fix is a reachable default;
        // this makes the term safe whatever the default is.
        const double lim = 0.10 * o_.target_kbps;
        err_local += std::clamp(
            o_.sdr_weight * (*o_.target_sdr_db - *frame_sdr_db), -lim, lim);
    }
    
    // Global error for Integral term (WITHOUT quality/SDR bias, ensuring perfect long-term convergence)
    const double err_global = o_.target_kbps - avg_global;
    
    // Use a longer leak memory (0.995) for the global rate error integration.
    // Freeze the integrator during silent frames to prevent windup.
    const double integral_leak_to_use = std::max(0.995, o_.integral_leak);
    // Frozen while the overflow ceiling binds and the error asks for more: the
    // content cannot use more, so accumulating that demand would only release
    // it as a burst of stuffing once the ceiling relaxes.
    const bool ovf_hold = ovf_binding_ && err_global > 0.0;
    if (!is_silent_frame_ && !ovf_hold) {
        integral_ = integral_leak_to_use * integral_ + err_global;
    } else {
        integral_ = integral_leak_to_use * integral_;
    }

    const double deriv_local = err_local - prev_err_;
    prev_err_ = err_local;
    last_err_ = err_local;

    const double u = o_.kp * err_local + o_.ki * integral_ + o_.kd * deriv_local;
    last_u_ = u;

    // See the note in quality mode: the gate's `silent` flag is a -63 dBFS
    // test meant for attack classification, and forcing 32 kbps on it wrecked
    // decay tails. The PID's own error term already gives a quiet frame fewer
    // bits, gradually, which is what a decay needs.
    const double target = o_.target_kbps + u;
    last_decision_kbps_ =
        o_.smoothing * target + (1.0 - o_.smoothing) * last_decision_kbps_;
    last_decision_kbps_ = std::clamp(last_decision_kbps_, o_.min_kbps, o_.max_kbps);
    // Deadband toward the target: snap small deviations exactly to the target
    // so the steady-state bitrate index (hence the HF cutoff) stops toggling
    // between adjacent steps frame-to-frame. Reduced to 3% to allow proper ABR tracking.
    constexpr double DEADBAND_FRAC = 0.03;
    if (std::abs(last_decision_kbps_ - o_.target_kbps)
            < DEADBAND_FRAC * o_.target_kbps)
        last_decision_kbps_ = o_.target_kbps;
    // After the deadband, or the snap to target would undo it.
    pre_ceiling_decision_kbps_ = last_decision_kbps_;
    ovf_binding_ = ovf_ceiling_kbps_ > 0.0 && last_decision_kbps_ > ovf_ceiling_kbps_;
    // Down to the *legal* minimum, not the ABR band's floor. The band floor
    // (target - 12) exists so a starved frame cannot be coded too coarsely;
    // a frame whose bits are overflowing the bank is by measurement not
    // starved. Clamping to the band floor is what made the ceiling inert under
    // ABR: at --abr 128 it could not go under 116, so the silence in
    // attack_from_silence kept producing 73 % stuffing.
    if (ovf_binding_)
        last_decision_kbps_ = std::max(ovf_ceiling_kbps_, legal_min_kbps_);
}

void AbrController::set_transient_hint(bool attack_from_silence,
                                       bool any_transient,
                                       bool all_silent) noexcept {
    is_silent_frame_ = all_silent;

    // An attack releases the overflow ceiling (see `observe`) at once.
    //
    // The ceiling is learned from overflow, which is a lagging signal: it
    // describes the frames that were just coded. An attack is a leading one,
    // and the lookahead already knows it is coming. Measured before this
    // release existed, at `-q 50` on the testset: after a silent run the
    // ceiling sat at the 32 kbps floor and relaxed only +5 % per frame, so
    // the attacks themselves were coded at 32 — transient_train's hole rate
    // went 4.75 % -> 40.66 % and attack_from_silence's segSNR 7.1 -> 0.5 dB.
    // Relying on the full bank to carry the attack does not work at low
    // rates, because the bank is tapered with the rate the ceiling just cut.
    if ((attack_from_silence || any_transient) && ovf_ceiling_kbps_ > 0.0) {
        if (ovf_binding_)
            last_decision_kbps_ = pre_ceiling_decision_kbps_;
        ovf_ceiling_kbps_ = 0.0;
        ovf_binding_ = false;
    }

    // The hints are *relative* boosts, not jumps to the clamps.
    //
    // They used to be absolute: `attack_from_silence` set the decision to
    // `max_kbps` outright and `any_transient` moved it halfway there. Both
    // discard whatever operating point the loop above had chosen, and in
    // quality mode that operating point is the entire meaning of `-q`. On
    // ordinary music the attack hint fires often enough that the rate lived at
    // the clamp: measured on `testset/mixed_music.wav`, every setpoint from
    // `-q 1` to `-q 60` produced the same 110.8 kbps — 60 % of the quality
    // scale was inert, and the encoder could not be asked for a low bitrate
    // through its primary control at all. With the same hints expressed
    // relatively the scale spans 35 -> 173 kbps monotonically.
    //
    // The pre-echo argument for the hint is about *this frame needing more
    // than its neighbours*, which is a ratio. It was never an argument for a
    // particular absolute rate, and at a 32 kbps operating point "spend
    // everything we can" meant 320 — a 10x excursion to cover one attack.
    //
    // Bounded multipliers keep the pre-echo behaviour (the attack still gets
    // the biggest share the loop can give it) while leaving the operating
    // point to whatever chose it. RMPS_HINT_LEGACY=1 restores the overrides.
    static const bool legacy = [] {
        const char* e = std::getenv("RMPS_HINT_LEGACY");
        return e && std::atoi(e) != 0;
    }();
    if (legacy) {
        if (attack_from_silence) {
            last_decision_kbps_ = o_.max_kbps;
        } else if (any_transient && !std::getenv("RMPS_NO_TRANSIENT_HINT")) {
            last_decision_kbps_ = std::min(
                o_.max_kbps,
                last_decision_kbps_ + 0.5 * (o_.max_kbps - last_decision_kbps_));
        } else if (all_silent && !std::getenv("RMPS_NO_SILENT_HINT")) {
            last_decision_kbps_ = std::max(
                32.0,
                last_decision_kbps_ - 0.5 * (last_decision_kbps_ - 32.0));
        }
        return;
    }

    /// Attack-from-silence: double the frame's rate. The worst pre-echo case,
    /// and the one the reservoir alone cannot fund at low rates.
    constexpr double AFS_GAIN     = 2.00;
    /// Ordinary transient: a third more.
    constexpr double TRANSIENT_GAIN = 1.35;
    /// True silence: half. The floor is the controller's own `min_kbps`, not a
    /// hardcoded 32 — at a 256 kbps operating point, dropping a silent frame
    /// to 32 is a rate excursion the reservoir then has to absorb.
    constexpr double SILENT_GAIN  = 0.50;

    double g = 1.0;
    if (attack_from_silence)
        g = AFS_GAIN;
    else if (any_transient && !std::getenv("RMPS_NO_TRANSIENT_HINT"))
        g = TRANSIENT_GAIN;
    else if (all_silent && !std::getenv("RMPS_NO_SILENT_HINT"))
        g = SILENT_GAIN;
    if (g > 1.0) {
        // Also keep the old *upper* bound — no further than halfway to the
        // clamp — so this stays a strict relaxation of the legacy behaviour at
        // rates where the operating point is already high. Without it a x1.35
        // on a 200 kbps decision reaches the clamp on every ordinary transient,
        // which measured +5 % of average rate at `--abr 192` on real material.
        last_decision_kbps_ = std::min(
            {last_decision_kbps_ * g,
             last_decision_kbps_ + 0.5 * (o_.max_kbps - last_decision_kbps_),
             o_.max_kbps});
    } else if (g < 1.0) {
        last_decision_kbps_ = std::max(last_decision_kbps_ * g, o_.min_kbps);
    }
}

AbrState AbrController::state() const noexcept {
    AbrState st {};
    st.target_kbps   = o_.target_kbps;
    st.avg_kbps      = last_avg_kbps_;
    st.decision_kbps = last_decision_kbps_;
    st.err           = last_err_;
    st.u             = last_u_;
    st.integral      = integral_;
    st.min_kbps      = o_.min_kbps;
    st.max_kbps      = o_.max_kbps;
    st.frames        = frames_;
    return st;
}

} // namespace rmps::control
