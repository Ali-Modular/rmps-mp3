#pragma once
// The encoder's circuit bends, as the stages that host them call them. The
// public description of each is in include/rmps/bend.hpp; this file is only
// where they plug in.
//
//   mdct::analyse        fold_source, butterflies, spectrum
//   iteration_loop       prune, starve, coded, account
//   decide_block_types   window
//   Encoder::Impl        begin_frame, end_frame

#include "rmps/bend.hpp"

#include <array>
#include <cstdint>

namespace rmps {
struct EncoderState;
}

namespace rmps::bend {

/// What the bends remember between frames. Lives in EncoderState.
struct State {
    /// Butterfly coefficients at `tables_alias`, rebuilt when the bend moves.
    std::array<std::int32_t, 8> ca{}, cs{};
    double tables_alias = 1.0;
    bool tables_valid = false;

    /// Per-channel held magnitude of every line, for smear and freeze.
    std::array<std::array<float, 576>, 2> held{};

    std::uint32_t rng = 0x2545F491u;

    /// Each long granule's coded spectrum, signed, in the domain it was coded
    /// in (L/R or M/S). Turned back into L/R and added to the view at the end
    /// of the frame, when both channels are in.
    std::array<std::array<std::array<float, 576>, 2>, 2> coded{};
    std::array<std::array<bool, 2>, 2> coded_valid{};

    BendView building{};   ///< this frame, as it accumulates
    BendView view{};       ///< the last whole frame
    int view_granules = 0;
};

/// Which analysed subband feeds subband `band` under the fold bend; -1 for
/// none (the mirror ran off the bottom of the spectrum).
int fold_source(int band, int fold) noexcept;

/// The alias-reduction coefficients for this frame. Points at the standard's
/// own tables (`std_ca`, `std_cs`) whenever the bend is at 1, so an unbent
/// encoder is bit-identical to an unmodified one.
void butterflies(EncoderState& s, const std::int32_t* std_ca, const std::int32_t* std_cs,
                 const std::int32_t*& ca, const std::int32_t*& cs);

/// Warp, smear and freeze, on one long-layout granule straight out of the
/// MDCT. Also records the before/after view. Short granules pass through.
void spectrum(EncoderState& s, int ch, int gr);

/// The mask bend: zeroes the scalefactor bands that stand too far below the
/// granule's most audible one. Long granules only; call after the masking
/// model has run.
void prune(EncoderState& s, int ch, int gr);

/// Records what a long granule was coded as, for the view's `after`. Call
/// once the granule's quantization and side info are final.
void coded(EncoderState& s, int ch, int gr, const int* ix);

/// The starve bend on one granule's grant.
int starve(const EncoderState& s, int max_bits) noexcept;

/// Records a granule's grant and spend for the view.
void account(EncoderState& s, int granted, int used) noexcept;

/// The window bend on the gate's choice.
int window(const EncoderState& s, int desired) noexcept;

void begin_frame(EncoderState& s) noexcept;
void end_frame(EncoderState& s) noexcept;

} // namespace rmps::bend
