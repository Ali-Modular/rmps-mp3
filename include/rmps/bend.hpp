#pragma once
#include <array>
#include <cstdint>

/// Circuit bends — deliberate, controllable faults in the coding path.
///
/// Everything else in this library exists to make the decoded stream sound as
/// close to the source as the bitrate allows. These do the opposite: each one
/// reaches into a single stage of the encoder or decoder and makes it do the
/// wrong thing on purpose, by an amount the caller sets. They are for using
/// the codec as an instrument, and they are all off by default — an encoder
/// or decoder whose bends are left at their defaults is bit-identical to one
/// built without them.
///
/// Every field may be changed between any two calls to encode()/decode(),
/// from the same thread; nothing is latched at construction, and nothing
/// needs a reset when a value moves.
namespace rmps {

/// Bends in the encoder, in signal order.
struct EncoderBends {
    /// ---- Polyphase filterbank ------------------------------------------
    /// Subband fold point, 1..32. Subbands at or above it are replaced by the
    /// mirror image of the ones below it — band `fold + n` carries band
    /// `fold - 1 - n` — so the spectrum under the fold is reflected up over
    /// it, each band landing in the wrong place and, for odd offsets, upside
    /// down. 32 leaves the filterbank alone.
    int fold = 32;

    /// ---- Alias-reduction butterflies -----------------------------------
    /// Multiplier on the rotation angle of the eight alias-reduction
    /// butterflies between neighbouring subbands. 1 is the standard's
    /// angle; 0 skips the stage, leaving the filterbank's aliasing in; past
    /// 1 the stage over-corrects, and negative values rotate the other way,
    /// smearing each band into its neighbour. Long blocks only, since the
    /// standard does not apply the stage to short ones.
    double alias = 1.0;

    /// ---- MDCT spectrum -------------------------------------------------
    /// Frequency-axis scale of the long-block spectrum. Line `i` is read from
    /// line `i / warp`, so 2 moves everything up an octave in 576-line steps
    /// (each source line used twice) and 0.5 down one (every other line
    /// dropped). No interpolation, no anti-aliasing: the result is the
    /// stepped, metallic shift a spectral resample gives. 1 is off.
    double warp = 1.0;

    /// Peak-hold decay of each spectral line's magnitude, 0..1. A line keeps
    /// the larger of its own magnitude and the previous granule's held
    /// magnitude times `smear`, so loud lines hang on and fade at this rate
    /// per granule (about 13 ms) while the signs stay live. 0 is off.
    double smear = 0.0;

    /// Hold every line's magnitude where it is. Signs keep coming from the
    /// live signal, so a frozen spectrum still moves — it is the colour that
    /// stops, not the sound. Releasing it carries on from the held values,
    /// decaying at `smear`.
    bool freeze = false;

    /// ---- Psychoacoustic model ------------------------------------------
    /// Masking-threshold gain in dB, applied as a prune: any scalefactor band
    /// whose energy falls under its masking threshold raised by this much is
    /// zeroed before quantization. Around 0 dB this removes only what the
    /// model already calls inaudible; above it the model is told far more is
    /// masked than is, and the spectrum breaks into the isolated, swirling
    /// partials of a starved stream at any bitrate. Negative infinity (the
    /// default) is off; anything below -60 dB prunes nothing in practice.
    double mask_db = -1.0e9;

    /// ---- Rate control --------------------------------------------------
    /// Fraction of each granule's bit grant the quantizer may actually
    /// spend, 0..1. The frame is still the bitrate's size; the bits not
    /// spent are banked in the reservoir until it is full and then padded,
    /// so this takes the stream below the lowest legal bitrate. 1 is off.
    double starve = 1.0;

    /// ---- Window switching ----------------------------------------------
    /// 0 lets the transient gate choose, 1 forces long blocks throughout
    /// (pre-echo smeared over every attack), 2 asks for short blocks on
    /// every granule (one third the frequency resolution, all the time).
    /// The standard's START/STOP transitions are still honoured, so the
    /// stream stays legal.
    int window = 0;
};

/// Bends in the decoder.
struct DecoderBends {
    /// Probability that each bit of a frame's main data — scalefactors and
    /// Huffman codes, not the header or side info — is flipped before it is
    /// decoded, 0..1. At 1e-5 an occasional chirp; at 1e-3 a stream dragged
    /// over a bad link. The header and side info are left intact so the
    /// stream keeps its framing however much is broken.
    double rot = 0.0;

    /// Exponent of the dequantizer's power law. The standard reconstructs a
    /// coded value `q` as `|q|^(4/3)`; this replaces 4/3 with `law`. Each
    /// granule is then rescaled to the energy 4/3 would have given it, so the
    /// bend moves energy between lines without changing the level: below 4/3
    /// quiet lines come up and loud ones down — a flattened, hissy spectrum;
    /// above it the loud lines take everything and the rest fall away.
    double law = 4.0 / 3.0;

    /// Decode the last `loop_frames` frames again, in order, for as long as
    /// this is set, instead of the incoming ones. The synthesis filterbank's
    /// overlap keeps running, so the loop has no click at its seam beyond the
    /// one the audio itself has. Incoming frames are still parsed, so the
    /// bit reservoir is in step when the loop is let go.
    bool loop = false;
    int loop_frames = 1;   ///< 1..8
};

/// What the encoder's bends did to the last frame, for drawing. Magnitudes
/// are the mean over granules and channels of |line|, normalised so a full-
/// scale sine's line reads about 1.
struct BendView {
    std::array<float, 576> before{};   ///< long-block spectrum as analysed
    std::array<float, 576> after{};    ///< the same after the spectral bends
    int bits_used = 0;                 ///< part2_3 bits over the frame
    int bits_granted = 0;              ///< before `starve` took its share
    bool short_blocks = false;         ///< any granule coded short
};

} // namespace rmps
