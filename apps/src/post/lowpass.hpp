#pragma once
namespace rmps { struct EncoderState;
namespace post {

double lowpass_side_cutoff_hz(int bitr_kbps);

/// Apply per-channel LPF: M=20 kHz, S=schedule based on current bitrate.
/// Zeros MDCT coefficients above the per-channel cutoff bin.
void apply_lowpass(EncoderState& s);

/// Boost MDCT coefficients above 14kHz by a configurable factor.
void apply_hf_gain_boost(EncoderState& s);

}}

