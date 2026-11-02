# Ali Modular fork: bends, a decoder, and real time

This fork turns rmps from a file encoder into a codec that can
sit in an audio signal path and be played. Three changes, all additive. With
every bend at its default and both new `Config` switches left alone, the
encoder is bit-identical to upstream.

1. **A decoder**: `rmps::Decoder`, `include/rmps/decoder.hpp`.
2. **Circuit bends** in the encoder and the decoder: `include/rmps/bend.hpp`.
3. **Two real-time switches** in `Config`: `fast_quantizer` and `bit_reservoir`.

## Why

Upstream rmps has no decoder, and at 2× real time (128 kbps, measured) it
can't keep up with an audio callback. The beam search is about 95% of its time.
The Ali Modular *Artifact* module needs a codec that runs at audio rate, with
the stages of the coding path exposed as controls.

The idea of bending a real MP3 encoder comes from
[MAIM](https://github.com/ArdenButterfield/Maim), which does it to LAME and
BladeEnc. None of MAIM's code is used here, and MAIM is GPL-3. The bends below
are designed for this encoder's own stages, which differ from LAME's:
fixed-point Q31 MDCT, a 3D masking model, outer_loop and beam quantizers, and a
PCM transient gate. Where a bend touches the same stage MAIM touches, it does
something different there, as the table shows.

## The decoder

`src/decode/decoder.cpp`: a floating-point MPEG-1 Layer III decoder, built to
ISO/IEC 11172-3. It covers everything this encoder emits: 32/44.1/48 kHz, mono,
stereo, M/S, long/start/stop/short blocks and the bit reservoir. It does not
cover MPEG-2 LSF, intensity stereo or mixed blocks, which this encoder never
produces. It builds its Huffman trees from the encoder's own tables, and its
synthesis window from the encoder's analysis window (D = 32·C), so the pair
cannot disagree. It skips the leading Xing/Info frame.

Round trip with `pre_gain_db = -1` (the encoder's own level compensation):
level ratio 1.002, 49–52 dB SNR on a pure tone at 128–320 kbps.

## The bends

| Bend | Stage | What it does | Where MAIM bends this stage, it… |
| --- | --- | --- | --- |
| `fold` | polyphase filterbank | Mirrors the subbands below a fold point up over it | …reassigns subbands by a user-drawn permutation |
| `alias` | alias-reduction butterflies | Scales each butterfly's *rotation angle* (0 = stage off, 1 = standard) | …replaces the butterfly with a free 2×2 matrix |
| `warp` | MDCT spectrum | Rescales the frequency axis (line *i* ← line *i*/warp) | …rotates the spectrum by a line offset |
| `smear`, `freeze` | MDCT spectrum | Peak-hold decay of each line's magnitude, live signs; freeze holds it | …blends magnitudes linearly with the previous frame's |
| `mask_db` | psychoacoustic model | Zeroes every band under its masking threshold raised by this much | …scales LAME's per-band thresholds |
| `starve` | rate control | Grants each granule only a fraction of its bits (with a floor) | (a similar squeeze on LAME's target bits) |
| `window` | transient gate | Forces long or short blocks, legal transitions kept | (no equivalent) |
| `rot` | decoder, main data | Random bit flips in scalefactors and Huffman data; header and side info left intact | (no decoder bends) |
| `law` | decoder, requantizer | Replaces the 4/3 power law, energy-normalised per granule | (no decoder bends) |
| `loop` | decoder, before filterbank | Replays the last 1–8 frames' spectra through the live filterbank | (no decoder bends) |

Hook points, all marked in the source:

- `src/core/mdct.cpp`: `fold`, `alias`, then `bend::spectrum` (`warp`, `smear`, `freeze`).
- `src/quant/iteration_loop.cpp`: `bend::prune` (`mask_db`), `bend::starve`, and `fast_quantizer`.
- `src/control/transient_gate.cpp`: `window`.
- `src/decode/decoder.cpp`: `rot`, `law`, `loop`.
- `src/bend/bend.cpp`: the encoder-side implementations, plus `BendView`, the per-frame spectra before and after the bends, for drawing.

## The real-time switches

- **`Config::fast_quantizer`**: every long granule goes to `outer_loop` (the global-gain bisection) instead of `beam_search`. At 128 kbps that's 2.2× → 36× real time. Quality drops to roughly shine's.
- **`Config::bit_reservoir`**: `false` sets `ResvMax = 0`, the same as `RMPS_NO_RESV=1` but as a setting. Every frame carries its own main data, so output stops lagging by up to five frames.

## Tested

From the Artifact repository: `tests/engine_test.cpp` covers timing,
`tests/module_test.cpp` covers the Rack module, and a bend sweep confirms each
bend changes the output, stays finite and keeps real time. Each bend at an
extreme setting, compared with the unbent output on the same input:

```
fold 12   -0.0 dB    alias 0   -10.5 dB   warp 2    +4.7 dB   smear .97  -0.9 dB
mask 20dB -0.0 dB    starve .1 -15.1 dB   window long -33.4 dB  window short -17.5 dB
rot 1e-4 -17.0 dB    rot 1e-2  -0.8 dB    law 1.0  -12.5 dB    law 1.8  -14.8 dB
```

All of them run at 26–35× real time with `fast_quantizer` on.
