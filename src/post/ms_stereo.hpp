#pragma once
namespace rmps { struct EncoderState;
namespace post {
/// L/R -> M/S in-place on MDCT coefficients (stereo only).
void apply_ms_stereo(EncoderState& s);
}}
