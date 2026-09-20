#pragma once

namespace rmps {
struct EncoderState;

namespace control {

/// Perceptual entropy of the frame sitting in `s.mdct_freq`, in bits.
///
/// This is the rate controller's *feed-forward* input: how many bits this
/// frame needs to put its quantization noise under the masking threshold. A
/// PID on measured SDR can only react after a frame has been coded, which is
/// far too slow for a release or a decay tail — by the time the loop notices
/// the frame was starved, the passage is over. `pe` is available before the
/// rate is chosen, so the rate can follow the demand with no lag at all.
///
/// Deliberately independent of `s.mpeg.bitr`: it is evaluated *before* the
/// bitrate exists for this frame, so anything rate-dependent would be
/// circular. That rules out `make_masking_params` (its `lf_strict_db` scales
/// with the rate) and the temporal pre/post-masking, which would also mutate
/// the real model's history. What is left — band energies, tonality, Bark
/// spreading, the absolute threshold of hearing — is the part that actually
/// carries the demand signal.
///
/// Bands the lowpass would discard need no special handling: they are
/// discarded precisely because they sit under the ATH, and a band under its
/// threshold contributes zero to `pe` by construction.
double frame_demand_bits(const EncoderState& s);

} // namespace control
} // namespace rmps
