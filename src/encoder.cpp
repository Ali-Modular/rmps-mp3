#include "rmps/rmps.hpp"

#include "control/abr_pid.hpp"
#include "control/demand.hpp"
#include "core/encoder_state.hpp"
#include "core/l3_bitstream.hpp"
#include "core/mdct.hpp"
#include "core/resample.hpp"
#include "core/zerophase_lpf.hpp"
#include "core/subband.hpp"
#include "core/tables.hpp"
#include "post/hf_adaptor.hpp"
#include "post/hf_extender.hpp"
#include "post/lowpass.hpp"
#include "quant/iteration_loop.hpp"
#include "quant/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace rmps {

namespace {

int find_samplerate_index(int freq) {
    for (int i = 0; i < tables::NUM_SAMPLERATES; ++i)
        if (tables::samplerates[i] == freq) return i;
    return -1;
}

int mpeg_version_for(int sr_index) {
    if (sr_index < 3) return 3;
    if (sr_index < 6) return 2;
    return 0;
}

int find_bitrate_index(int kbps, int version) {
    for (int i = 0; i < tables::NUM_BITRATES; ++i)
        if (tables::bitrates[i][version] == kbps) return i;
    return -1;
}

int find_nearest_bitrate_index(int kbps, int version, int& snapped) {
    int best_idx = 1;
    int best_diff = 1000000;
    int best_kbps = 0;
    for (int i = 1; i < tables::NUM_BITRATES - 1; ++i) {
        const int b = tables::bitrates[i][version];
        if (b < 0) continue;
        const int diff = std::abs(kbps - b);
        if (diff < best_diff) {
            best_diff = diff;
            best_idx  = i;
            best_kbps = b;
        }
    }
    snapped = best_kbps;
    return best_idx;
}

int granules_per_frame_for(int version) {
    return version == 3 ? 2 : 1;
}

} // namespace

/// Coded sample rate for an input rate and operating bitrate.
///
/// Only ever downward, only ever to another MPEG-1 rate, and only at rates
/// where the Mid lowpass floor is already far below the new Nyquist — see the
/// note on `Config::coded_samplerate`. Returns `in_rate` when there is nothing
/// to gain, so the resampler degenerates to a copy.
int auto_coded_samplerate(int in_rate, int kbps, int channels) {
    // 48 kbps, measured, not assumed. A/B over the testset at 32/48/64 kbps
    // (`RMPS_CODED_SR=0` against `=32000`, one binary):
    //
    //   segSNR     32k     48k     64k
    //   mixed_music      +4.36   +1.93   -0.75
    //   harmonic_tone    +3.83   +4.79   +0.64
    //   two_tone_poly    +3.12   +2.41   -0.60
    //   transient_train  +0.83   +1.33   +0.23
    //
    //   hole_rate: 15 clips better / 4 worse overall, mean -4.78 points.
    //   stereo_decorrelated@32k 97.2 % -> 15.0 %, hf_hiss_decay@48k
    //   33.8 % -> 17.4 %, transient_train@32k 35.9 % -> 27.5 %.
    //
    // By 64 kbps it is a wash and turns negative on the clips with the most
    // content, so the threshold sits under it rather than on it.
    //
    // What it costs is real and shows up exactly where it should: `sweep`
    // loses 18-23 dB of segSNR because it glides past 16 kHz and a 32 kHz
    // stream cannot carry that at all, and `hf_hiss_decay` loses 4-6 dB of
    // hf_balance for the same reason. Material with genuine content above
    // 16 kHz gives it up — at 48 kbps that content was being coded at -50 dB
    // anyway (see `post/lowpass.cpp`), which is why the trade comes out this
    // far ahead on everything that is not a test tone.
    //
    // Mono's side info is 17 bytes against stereo's 32, so the same argument
    // applies at a lower rate; untested, so it stays conservative.
    const int thr = (channels >= 2) ? 48 : 40;
    if (kbps <= 0 || kbps > thr) return in_rate;
    if (in_rate == 44100 || in_rate == 48000) return 32000;
    return in_rate;
}

/// **The automatic policy is off.** `auto_coded_samplerate` above is kept, and
/// its measurements with it, but resampling now happens only when asked for —
/// `Config::coded_samplerate` or `RMPS_CODED_SR`.
///
/// What the numbers above leave out is pre-echo, and it is the deciding term.
/// A 32 kHz stream codes the same 576-line window over 36 ms of time instead
/// of 26.1, so an attack's quantization noise is smeared over 38 % longer.
/// Measured at 48 kbps, isolating the sample rate (`RMPS_CODED_SR=0` against
/// the policy, everything else equal):
///
///     clip                  pre_echo_db      segSNR      hole_rate
///     attack_from_silence  -66.1 -> -34.8   +0.29       3.94 -> 4.55 %
///     transient_train      -46.0 -> -38.3   +1.34      22.95 -> 15.41 %
///
/// 31 dB of pre-echo on the attack-from-silence path is not a trade worth the
/// hole rate, and it is audible as the thing users describe as a click added
/// just before a hit — the same artefact `short_block_hf_threshold` was just
/// lowered to reduce. Shipping both at once would have handed back more than
/// that change won.
///
/// It is a real gain on sustained material (see the table above and
/// `Config::coded_samplerate`), so it stays available: set
/// `RMPS_CODED_SR=32000` for program material without sharp transients, where
/// the hole-rate and segSNR wins are large and there is no attack to smear.
/// Turning it on by default needs the window-length cost paid for first —
/// short blocks that actually fire on real material would be the way.
int coded_samplerate_policy(int in_rate, int /*kbps*/, int /*channels*/) {
    return in_rate;
}

struct Encoder::Impl {
    EncoderState               state;
    std::vector<std::uint8_t>  last_output;
    FrameStats                 last_stats;
    std::unique_ptr<control::AbrController> abr;

    // ----- Input resampling (see Config::coded_samplerate) -----
    std::unique_ptr<core::Resampler> resampler;
    /// Resampled interleaved PCM not yet consumed as a whole coded frame.
    std::vector<std::int16_t>  rs_pending;
    /// Bytes accumulated across the 0..N coded frames one `encode()` call can
    /// now produce. Without resampling this holds exactly one frame, as before.
    std::vector<std::uint8_t>  out_accum;

    // ----- Zero-phase input lowpass (see Config::prefilter_lowpass) -----
    std::unique_ptr<core::ZeroPhaseLowpass> pre_lpf;
    std::vector<std::int16_t>  lpf_out;

    // ----- Online high-frequency bandwidth extension -----
    std::unique_ptr<post::HFExtender> hf_extender;

    /// Whether input has to be buffered and framed here rather than passed
    /// through one chunk per frame: either stage can release a different
    /// number of samples than it was given on any one call.
    bool buffered() const {
        return pre_lpf || (resampler && !resampler->passthrough());
    }

    /// Code every whole frame waiting in `rs_pending`.
    void drain_frames() {
        const std::size_t ch = static_cast<std::size_t>(state.wave.channels);
        const std::size_t need = samples_per_pass() * ch;
        std::size_t pos = 0;
        while (rs_pending.size() - pos >= need) {
            encode_one(rs_pending.data() + pos, need);
            pos += need;
        }
        if (pos)
            rs_pending.erase(rs_pending.begin(),
                             rs_pending.begin() + static_cast<std::ptrdiff_t>(pos));
    }

    /// Hand stage-one output (filtered or raw PCM) to the resampler, or
    /// straight to framing when the stream is coded at the input rate.
    void to_framing(const std::int16_t* data, std::size_t samples) {
        // RMPS_PRE_LPF_DUMP=<path>: raw interleaved int16 of what framing
        // receives, for checking the lowpass against an offline filtfilt.
        if (pre_lpf) {
            static std::FILE* dump = [] {
                const char* e = std::getenv("RMPS_PRE_LPF_DUMP");
                return e ? std::fopen(e, "wb") : nullptr;
            }();
            if (dump && samples) {
                std::fwrite(data, sizeof(std::int16_t), samples, dump);
                std::fflush(dump);
            }
        }
        const std::size_t ch = static_cast<std::size_t>(state.wave.channels);
        if (resampler && !resampler->passthrough())
            resampler->process(data, samples / ch, rs_pending);
        else
            rs_pending.insert(rs_pending.end(), data, data + samples);
    }
    int input_samplerate = 0;   ///< the caller's rate, for `samples_per_pass`

    // ----- Xing/Info (VBR) tag -----
    std::vector<std::uint32_t> frame_sizes;   ///< bytes per produced frame
    std::vector<std::uint8_t>  tag_frame;
    bool tag_prepended = false;

    /// Build the Xing ("Xing" for ABR, "Info" for CBR) tag frame. With
    /// `final_pass` false this is the zero-count placeholder emitted at
    /// stream start; with true it carries the real frame/byte counts and
    /// the 100-point seek TOC. Both passes produce the same frame size,
    /// so the finalized tag can overwrite the placeholder in place.
    void build_vbr_tag(bool final_pass) {
        const int sr         = state.wave.samplerate;
        const int side_bytes = state.sideinfo_len / 8;   // incl. 4-byte header
        const int need       = side_bytes + 120;         // magic..TOC..quality
        int bi = -1, fsize = 0;
        for (int i = 1; i < tables::NUM_BITRATES - 1; ++i) {
            const int br = tables::bitrates[i][state.mpeg.version];
            if (br < 0) continue;
            const int sz = 144 * br * 1000 / sr;
            if (sz >= need) { bi = i; fsize = sz; break; }
        }
        if (bi < 0) { tag_frame.clear(); return; }

        tag_frame.assign(static_cast<std::size_t>(fsize), 0u);
        const auto& m = state.mpeg;
        tag_frame[0] = 0xFF;
        tag_frame[1] = static_cast<std::uint8_t>(
            0xE0 | (m.version << 3) | (m.layer << 1) | (m.crc ? 0 : 1));
        tag_frame[2] = static_cast<std::uint8_t>(
            (bi << 4) | ((m.samplerate_index % 3) << 2) | (m.ext));
        tag_frame[3] = static_cast<std::uint8_t>(
            (static_cast<int>(m.mode) << 6) | (m.mode_ext << 4)
            | ((m.copyright ? 1 : 0) << 3) | ((m.original ? 1 : 0) << 2)
            | static_cast<int>(m.emph));

        std::size_t p = static_cast<std::size_t>(side_bytes);
        const char* magic = abr ? "Xing" : "Info";
        std::memcpy(tag_frame.data() + p, magic, 4); p += 4;
        auto put_be32 = [&](std::uint32_t v) {
            tag_frame[p++] = static_cast<std::uint8_t>(v >> 24);
            tag_frame[p++] = static_cast<std::uint8_t>(v >> 16);
            tag_frame[p++] = static_cast<std::uint8_t>(v >> 8);
            tag_frame[p++] = static_cast<std::uint8_t>(v);
        };
        put_be32(0x00000007u);   // FRAMES | BYTES | TOC
        if (!final_pass) return; // counts and TOC stay zero in the placeholder

        std::uint64_t audio_bytes = 0;
        for (const auto fs : frame_sizes) audio_bytes += fs;
        const std::uint64_t total_bytes = audio_bytes + tag_frame.size();
        const std::uint32_t n_frames =
            static_cast<std::uint32_t>(frame_sizes.size());
        put_be32(n_frames);
        put_be32(static_cast<std::uint32_t>(total_bytes));

        // TOC: byte offset (scaled to 0..255 of the whole file) at each
        // percent of stream time. Frames are equal-duration, so percent of
        // time == percent of frame index.
        std::vector<std::uint64_t> cum(frame_sizes.size() + 1, 0);
        for (std::size_t i = 0; i < frame_sizes.size(); ++i)
            cum[i + 1] = cum[i] + frame_sizes[i];
        for (int t = 0; t < 100; ++t) {
            const std::size_t idx =
                (static_cast<std::size_t>(t) * frame_sizes.size()) / 100;
            const std::uint64_t off = tag_frame.size() + cum[idx];
            std::uint64_t v = (total_bytes > 0)
                ? (off * 256u) / total_bytes : 0u;
            if (v > 255u) v = 255u;
            tag_frame[p++] = static_cast<std::uint8_t>(v);
        }
    }
    Config                     cfg;

    bool init(const Config& c) {
        cfg = c;
        input_samplerate = c.samplerate;
        const int initial_kbps =
            c.quality_sdr_db ? 128 : c.abr_target_kbps.value_or(c.bitrate);

        // Rate the stream is *expected* to operate at, for decisions that must
        // be made once and cannot be revisited — the coded sample rate is one,
        // since a single MP3 stream cannot change it mid-file.
        //
        // In quality mode `initial_kbps` is a seed of 128 that says nothing
        // about where the loop will settle, so use the setpoint instead. The
        // mapping is the same pinned-rate measurement `-q` is calibrated on:
        // real material reads 7.0-11.2 dB of Fullness/Bleedless at 32 kbps.
        // Only the very bottom of the scale is claimed — at `-q 10` the four
        // 30 s excerpts landed at 35-50 kbps, and by `-q 20` the spread is
        // 45-74, which is already past where 32 kHz coding pays. Material
        // varies enough that a tighter prediction would be guessing.
        const double expected_kbps =
            c.quality_sdr_db
                ? (*c.quality_sdr_db <= 10.5 ? 48.0 : 128.0)
                : static_cast<double>(initial_kbps);

        // Coded rate: explicit config wins, then RMPS_CODED_SR (so the
        // decision can be A/B'd on one binary), then the automatic policy.
        int coded_sr = c.coded_samplerate;
        if (coded_sr <= 0) {
            if (const char* e = std::getenv("RMPS_CODED_SR")) {
                const int v = std::atoi(e);
                // 0 disables the automatic policy rather than re-enabling it,
                // so a sweep can pin "never resample" as well as a rate.
                coded_sr = (v > 0) ? v : c.samplerate;
            } else {
                coded_sr = coded_samplerate_policy(
                    c.samplerate, static_cast<int>(expected_kbps), c.channels);
            }
        }

        const int sr_idx = find_samplerate_index(coded_sr);
        if (sr_idx < 0) return false;
        const int ver    = mpeg_version_for(sr_idx);
        if (ver != 3) {
            // MPEG-2 / 2.5 needs the LSF scalefactor scheme, which is not
            // implemented — see Config::coded_samplerate. Refuse rather than
            // emit a stream whose scalefactors the decoder will misread.
            if (coded_sr != c.samplerate) return false;
        }
        resampler = std::make_unique<core::Resampler>(c.samplerate, coded_sr,
                                                      c.channels);
        rs_pending.clear();

        // Zero-phase input lowpass. The cutoff follows the *coded* rate — the
        // bandwidth the stream will actually carry — and the filter runs at the
        // input rate, ahead of the resampler. Skipped where the cutoff would sit
        // at or above 95 % of the input's Nyquist, since there is nothing left
        // to remove there.
        {
            bool on = c.prefilter_lowpass;
            if (const char* e = std::getenv("RMPS_PRE_LPF")) on = std::atoi(e) != 0;
            const double fc = (coded_sr <= 32000) ? 14000.0 : 20000.0;
            int order = 4;
            if (const char* e = std::getenv("RMPS_PRE_LPF_ORDER")) order = std::atoi(e);
            pre_lpf.reset();
            lpf_out.clear();
            if (on && fc < 0.95 * 0.5 * c.samplerate)
                pre_lpf = std::make_unique<core::ZeroPhaseLowpass>(
                    c.samplerate, fc, c.channels, order);
        }
        
        int snapped_kbps = 0;
        int br_idx = -1;
        if (c.abr_target_kbps || c.quality_sdr_db) {
            br_idx = find_nearest_bitrate_index(initial_kbps, ver, snapped_kbps);
        } else {
            br_idx = find_bitrate_index(initial_kbps, ver);
            snapped_kbps = initial_kbps;
        }
        if (br_idx < 0) return false;

        state.wave.channels   = c.channels;
        // Everything downstream — filterbank, band tables, psychoacoustics,
        // frame timing — works at the *coded* rate. The input rate survives
        // only in `input_samplerate`, for sizing the caller's chunks.
        state.wave.samplerate = coded_sr;
        state.mpeg.bitr       = snapped_kbps;
        state.mpeg.emph       = c.emph;
        state.mpeg.copyright  = c.copyright;
        state.mpeg.original   = c.original;
        state.mpeg.samplerate_index   = sr_idx;
        state.mpeg.version            = ver;
        state.mpeg.bitrate_index      = br_idx;
        state.mpeg.granules_per_frame = granules_per_frame_for(ver);
        state.mpeg.bits_per_slot      = 8;
        state.mpeg.layer              = 1;

        if (state.wave.channels == 2) {
            state.mpeg.mode     = Mode::JointStereo;
            state.mpeg.mode_ext = 2;
        } else {
            state.mpeg.mode     = Mode::Mono;
            state.mpeg.mode_ext = 0;
        }

        const double avg_slots =
            (state.mpeg.granules_per_frame * 576.0 / state.wave.samplerate)
            * (1000.0 * state.mpeg.bitr / state.mpeg.bits_per_slot);
        state.mpeg.whole_slots_per_frame = static_cast<int>(avg_slots);
        state.mpeg.frac_slots_per_frame  = avg_slots - state.mpeg.whole_slots_per_frame;
        state.mpeg.slot_lag              = -state.mpeg.frac_slots_per_frame;
        if (state.mpeg.frac_slots_per_frame == 0.0) state.mpeg.padding = 0;

        if (state.mpeg.granules_per_frame == 2)
            state.sideinfo_len = 8 * (state.wave.channels == 1 ? (4 + 17) : (4 + 32));
        else
            state.sideinfo_len = 8 * (state.wave.channels == 1 ? (4 +  9) : (4 + 17));

        subband::initialise(state);
        mdct::initialise(state);
        quant::initialise(state);

        state.hf_adaptor = std::make_unique<post::HFAdaptor>(
            6000.0, c.hf_max_atten_m_db, c.hf_max_atten_s_db, 25.0, 0.45);

        // --- Search effort ---
        // `effort`, `beam_width`, `gain_window` and `sf_max_candidate` were
        // dead config: beam_search.cpp hard-coded its own constants and never
        // read them, so --effort changed nothing. Map them linearly with
        // effort 7 (the default) landing exactly on those old constants, so
        // the default encode is unchanged and the knob now has an effect.
        {
            const int e = std::clamp(c.effort, 0, 10);
            auto scaled = [e](int at_7, int lo, int hi) {
                return std::clamp((e * at_7 + 3) / 7, lo, hi);
            };
            // Floor at 3, not 0. `gain_window` is the only thing deciding
            // how far the search may move from the seed gain, and if no gain
            // in that window fits the real budget the granule falls back to
            // the legacy "coarsen everything with sf=0" path, where none of
            // the beam's per-band allocation applies. Measured on real program
            // material at 128 kbps: a window of 1 (what effort 2 derived) fell
            // back on 51 % of granule-channels; a window of 3 falls back on
            // 4 %. The old floor made the fast settings fast by disabling the
            // search, not by speeding it up.
            state.cfg_gain_window   = scaled(4, 3, 6);
            // Capped at 5: the per-(sfb, sf) candidate cache is a stack array
            // sized by this, and 6 candidates per band is already ~130 KB.
            state.cfg_sf_max        = scaled(5, 1, 5);
            state.cfg_rate_probes   = scaled(8, 1, 16);
            state.cfg_rate_buckets  = scaled(64, 8, 128);
            state.cfg_polish_iters  = scaled(6, 0, 12);
            if (c.beam_width      > 0) state.cfg_rate_probes = std::min(c.beam_width, 16);
            if (c.gain_window     >= 0) state.cfg_gain_window = std::min(c.gain_window, 6);
            if (c.sf_max_candidate > 0) state.cfg_sf_max      = std::min(c.sf_max_candidate, 5);
            // Measurement override for the scalefactor range (1..5). The CLI
            // does not expose `effort`, and the range is what decides whether
            // a quiet band next to a loud one can be kept at all.
            if (const char* e = std::getenv("RMPS_SF_MAX")) {
                const int v = std::atoi(e);
                if (v >= 1 && v <= 5) state.cfg_sf_max = v;
            }
        }

        state.cfg_mask_alpha    = c.mask_alpha;
        if (const char *e = std::getenv("RMPS_MASK_ALPHA"))
            state.cfg_mask_alpha = std::atof(e);
        state.cfg_mask_lambda_R = c.mask_lambda_R;
        state.cfg_lf_strict_db        = c.lf_strict_db;
        state.cfg_lf_strict_cutoff_hz = c.lf_strict_cutoff_hz;
        state.pre_gain          = std::pow(10.0, c.pre_gain_db / 20.0);
        state.hf_weight_boost   = std::max(1.0, c.hf_weight_boost);
        state.cfg_hf_zero_boost_db = std::clamp(c.hf_zero_boost_db, 0.0, 24.0);
        if (const char *e = std::getenv("RMPS_ZERO_BOOST"))
            state.cfg_hf_zero_boost_db = std::clamp(std::atof(e), 0.0, 24.0);
        state.cfg_noise_shaping_alpha = std::clamp(c.noise_shaping_alpha, 0.0, 0.95);
        // RMPS_NSA scales the shaper without switching it off (RMPS_NO_NSA=1
        // does that). The value seeds the tonality heuristic and the per-band
        // alpha ladder, so it moves the whole axis rather than pinning it.
        if (const char *e = std::getenv("RMPS_NSA"))
            state.cfg_noise_shaping_alpha = std::clamp(std::atof(e), 0.0, 0.95);
        state.cfg_waveform_reg  = c.waveform_reg;
        if (const char *e = std::getenv("RMPS_WAVEFORM_REG"))
            state.cfg_waveform_reg = std::atof(e);
        state.cfg_ms_atten_min_db = std::clamp(c.ms_atten_min_db, 0.0, 36.0);
        state.cfg_ms_atten_max_db = std::clamp(c.ms_atten_max_db, 0.0, 48.0);
        state.cfg_adaptive_ms   = c.adaptive_joint_stereo;
        state.ms_prev_frame     = true;
        state.cfg_block_switching = c.block_switching;
        state.cfg_fast_quantizer = c.fast_quantizer;
        state.prev_block_type   = {BLOCK_LONG, BLOCK_LONG};
        {
            double hf_thr = c.short_block_hf_threshold;
            if (const char *e = std::getenv("RMPS_SB_HF_THR"))
                hf_thr = std::atof(e);
            state.transient_gate.set_hf_transient_threshold(hf_thr);
        }

        // --- High-Frequency Extension ---
        state.cfg_hf_extension = c.hf_extension;
        if (const char *e = std::getenv("RMPS_HF_EXT"))
            state.cfg_hf_extension = (std::atoi(e) != 0);
        state.cfg_hf_ext_cutoff_hz = c.hf_ext_cutoff_hz;
        if (const char *e = std::getenv("RMPS_HF_CUTOFF"))
            state.cfg_hf_ext_cutoff_hz = std::atof(e);
        state.cfg_hf_ext_obj_gain_db = c.hf_ext_obj_gain_db;
        state.cfg_hf_ext_res_gain_db = c.hf_ext_res_gain_db;
        state.cfg_hf_ext_tilt_db_oct = c.hf_ext_tilt_db_oct;

        if (state.cfg_hf_extension) {
            hf_extender = std::make_unique<post::HFExtender>();
            post::HFExtenderConfig hf_cfg;
            hf_cfg.cutoff_hz = state.cfg_hf_ext_cutoff_hz;
            hf_cfg.max_out_hz = 20000.0;
            hf_cfg.obj_gain_db = state.cfg_hf_ext_obj_gain_db;
            hf_cfg.res_gain_db = state.cfg_hf_ext_res_gain_db;
            hf_cfg.tilt_db_oct = state.cfg_hf_ext_tilt_db_oct;
            hf_extender->init(hf_cfg, state.wave.samplerate, state.wave.channels);
        }

        // Cross-frame bit reservoir (main_data_begin). ResvSize/ResvMax plan
        // the banking (reservoir::adjust / frame_end keep everything above
        // ResvMax stuffed and the bank byte-aligned); l3_bitstream carries
        // the authoritative BYTE accounting at emission. The cap must stay
        // under the main_data_begin field range (511 bytes MPEG-1, 255
        // MPEG-2) with margin.
        state.ResvMax  = (ver == 3) ? 8 * 440 : 8 * 200;
        // RMPS_NO_RESV=1 disables cross-frame banking entirely: every frame
        // then carries its own main data and main_data_begin stays at 0.
        // Bisection aid — the reservoir is the one layer whose correctness
        // the side-info audits cannot check, because they only prove the
        // encoder's own accounting is self-consistent.
        if (const char *e = std::getenv("RMPS_NO_RESV"))
            if (std::atoi(e) != 0) state.ResvMax = 0;
        if (!c.bit_reservoir) state.ResvMax = 0;
        state.ResvSize = 0;
        // Reset the cross-frame reservoir/FIFO state for this stream.
        state.md_fifo.clear();
        state.md_emitted = 0;
        state.resv_bytes = 0;
        state.pending_frames.clear();

        if (c.quality_sdr_db) {
            // Quality-targeted VBR: the SDR is the setpoint and the rate is
            // free. Seeded mid-range so the loop converges from either side.
            control::AbrController::Options o;
            o.quality_mode  = true;
            // The `-q` scale is calibrated in SDR. Fullness/Bleedless uses the
            // same units but reads about 2.8 dB higher on the same encode —
            // it reports the worse of two signal-to-failure ratios where SDR
            // reports one signal-to-total-error ratio, and the total is
            // always the larger denominator. Offset the setpoint so a given
            // `-q` keeps producing the bitrate it used to; without this,
            // `-q 50` dropped from 220 to 193 kbps purely from the change of
            // yardstick. Measured by matching rates: FB q57 = SDR q50.
            //
            // Now 0. Two things underneath it changed: the frame score is an
            // energy-weighted ratio of summed powers instead of a mean of
            // per-granule dB, and the `-q` endpoints were re-derived from
            // pinned-rate FB measurements on real material (see main.cpp). The
            // scale is already expressed in the units this controller holds,
            // so an offset calibrated against the old aggregation would only
            // move it back off.
            constexpr double FB_MINUS_SDR_DB = 0.0;
            const char* e_sdr = std::getenv("RMPS_Q_SDR");
            const bool use_fb = !(e_sdr && std::atoi(e_sdr) != 0);
            o.target_sdr_db =
                *c.quality_sdr_db + (use_fb ? FB_MINUS_SDR_DB : 0.0);
            o.target_kbps   = 128.0;
            o.kp = 0.50; o.ki = 0.05; o.kd = 0.10;
            o.smoothing     = 0.25;
            o.mpeg_version  = ver;
            abr = std::make_unique<control::AbrController>(o);
        } else if (c.abr_target_kbps) {
            control::AbrController::Options o;
            o.target_kbps = *c.abr_target_kbps;
            if (c.target_sdr_db) o.target_sdr_db = c.target_sdr_db;
            if (c.abr_kp) o.kp = *c.abr_kp;
            if (c.abr_ki) o.ki = *c.abr_ki;
            if (c.abr_kd) o.kd = *c.abr_kd;
            o.min_kbps = c.abr_min_kbps;
            o.max_kbps = c.abr_max_kbps;
            o.mpeg_version = ver;
            abr = std::make_unique<control::AbrController>(o);
        }
        return true;
    }

    std::size_t samples_per_pass() const {
        return static_cast<std::size_t>(state.mpeg.granules_per_frame) * 576u;
    }

    /// Encode exactly one coded frame's worth of interleaved PCM, appending
    /// its bytes to `out_accum`.
    ///
    /// One-frame lookahead. To know what is coming the encoder has to hold the
    /// incoming frame back and encode the previous one, so this emits frame
    /// N−1 and the first call emits nothing.
    void encode_one(const std::int16_t* data, std::size_t n) {
        auto& st = state;
        const bool have_staged = staged_valid;

        // Demand probe on the *incoming* frame, before it is staged. Stateless
        // on purpose: the transient gate carries history across frames, so
        // running it twice per frame — once to look ahead, once for real —
        // would corrupt the very decisions it exists to make.
        next_frame_hint = probe_demand(PcmChunk{data, n});

        if (!have_staged) {
            staged.assign(data, data + n);
            staged_valid = true;
            last_output.clear();
            return;
        }

        // Encode the *staged* frame while `data` (its successor) supplies the
        // lookahead. `staged` must not be overwritten until after this runs.
        st.buffer[0]     = staged.data();
        st.buffer_idx[0] = 0;
        if (st.wave.channels == 2) {
            st.buffer[1]     = staged.data();
            st.buffer_idx[1] = 1;
        }
        st.lookahead_valid      = true;
        st.lookahead_attack     = next_frame_hint.attack;
        st.lookahead_attack_gr0 = next_frame_hint.attack_gr0;
        st.lookahead_demand     = next_frame_hint.demand;

        run_frame(st.wave.channels);
        copy_output();
        out_accum.insert(out_accum.end(), last_output.begin(),
                         last_output.end());

        staged.assign(data, data + n);
        staged_valid = true;
    }

    // ---- One-frame lookahead ------------------------------------------
    std::vector<std::int16_t> staged;      ///< PCM held back by one frame
    bool staged_valid = false;
    struct DemandHint {
        bool   attack = false; ///< a sharp energy step lands in this frame
        /// ...and it lands in the frame's *first* granule. This is the case
        /// the block-type decision cannot see on its own: the START window
        /// that has to precede a SHORT belongs to the previous frame, which
        /// was already committed. See `decide_block_types`.
        bool   attack_gr0 = false;
        double demand = 0.0;   ///< its energy relative to the frame before
    };
    DemandHint next_frame_hint{};
    double prev_probe_energy = 0.0;

    /// Cheap, stateless demand probe on raw PCM.
    ///
    /// The reservoir's one real question is "will the next frame want more
    /// bits than its flat share", and the expensive answer (masking → `pe`)
    /// is not available for a frame that has not been through the MDCT yet.
    /// The cheap answer is the same one the transient gate starts from: sharp
    /// energy steps. Split the frame into the gate's sub-windows, take the
    /// largest step between consecutive sub-windows, and compare the frame's
    /// own energy to the previous probe's.
    ///
    /// This deliberately does not touch `state.transient_gate`: that object
    /// carries history between frames, and analysing a frame twice through it
    /// would make its own attack decisions depend on how far ahead we looked.
    DemandHint probe_demand(const PcmChunk& pcm) {
        DemandHint h;
        const int ch = state.wave.channels;
        if (!pcm.data || pcm.num_samples == 0 || ch <= 0)
            return h;
        constexpr int kSubs = 8;
        const std::size_t frames = pcm.num_samples / static_cast<std::size_t>(ch);
        const std::size_t per_sub = frames / kSubs;
        if (per_sub == 0)
            return h;
        double sub_e[kSubs] = {};
        // High-frequency content per sub-window, on the same first-difference
        // vs boxcar basis the transient gate classifies attacks with. The
        // energy step alone is far too loose to drive a *coding* decision —
        // it fires on ordinary music, and using it to place short blocks
        // measured 20 eval cells worse (mixed_music@48 29.5 -> 9.9 dB). The
        // gate only asks for a short block when the attack is HF-rich, so the
        // lookahead has to apply the same test or it proposes short blocks
        // the gate itself would have refused.
        double sub_hf[kSubs] = {};
        double sub_lf[kSubs] = {};
        constexpr int kBox = 16;
        for (int sIdx = 0; sIdx < kSubs; ++sIdx) {
            double acc = 0.0, hf = 0.0, lf = 0.0, box = 0.0;
            const std::size_t base = static_cast<std::size_t>(sIdx) * per_sub;
            double prev = 0.0;
            std::vector<double> hist(per_sub, 0.0);
            for (std::size_t i = 0; i < per_sub; ++i) {
                double m = 0.0;
                for (int c = 0; c < ch; ++c)
                    m += pcm.data[(base + i) * static_cast<std::size_t>(ch) +
                                  static_cast<std::size_t>(c)];
                m /= ch;
                acc += m * m;
                if (i > 0) { const double d = m - prev; hf += d * d; }
                prev = m;
                hist[i] = m;
                box += m;
                if (i >= static_cast<std::size_t>(kBox)) box -= hist[i - kBox];
                if (i + 1 >= static_cast<std::size_t>(kBox)) {
                    const double avg = box / kBox;
                    lf += avg * avg;
                }
            }
            sub_e[sIdx] = acc / static_cast<double>(per_sub);
            sub_hf[sIdx] = hf;
            sub_lf[sIdx] = std::max(lf, 1.0);
        }
        double frame_e = 0.0;
        double worst_step = 1.0;
        int worst_sub = -1;
        for (int sIdx = 0; sIdx < kSubs; ++sIdx) {
            frame_e += sub_e[sIdx];
            const double prev =
                (sIdx > 0) ? sub_e[sIdx - 1] : std::max(prev_probe_energy / kSubs, 1.0);
            const double step = sub_e[sIdx] / std::max(prev, 1.0);
            if (step > worst_step) { worst_step = step; worst_sub = sIdx; }
        }
        frame_e /= kSubs;
        // 8× (≈9 dB) between neighbouring sub-windows is the same order the
        // transient gate treats as an attack; this only has to be roughly
        // right, since it steers a bit *grant*, not a coding decision.
        h.attack = (worst_step >= 8.0);
        // The sub-windows split the frame evenly, so the first half belongs to
        // granule 0.
        // Same HF-richness gate the transient gate applies before asking for
        // a short block (`short_block_hf_threshold`), so the lookahead cannot
        // propose one the gate would refuse.
        static const double hf_thr = [] {
            if (const char *e = std::getenv("RMPS_SB_HF_THR"))
                return std::atof(e);
            return 0.4;
        }();
        const bool hf_rich =
            worst_sub >= 0 &&
            (sub_hf[worst_sub] / sub_lf[worst_sub]) >= hf_thr;
        // Which MDCT granule the step lands in is not simply which half of
        // the incoming PCM holds it: the polyphase analysis window carries
        // ~481 samples of delay, so PCM from the first half of a frame is
        // still being folded into the *second* granule's transform. Selecting
        // on the raw first half places the START a granule early, which is
        // worse than the late placement it was meant to fix.
        static const int half = [] {
            if (const char *e = std::getenv("RMPS_LA_HALF"))
                return std::atoi(e);
            return 1;    // second half of the incoming PCM
        }();
        const bool in_gr0 = (half == 0) ? (worst_sub < kSubs / 2)
                                        : (worst_sub >= kSubs / 2);
        h.attack_gr0 = h.attack && hf_rich && worst_sub >= 0 && in_gr0;
        h.demand = frame_e / std::max(prev_probe_energy, 1.0);
        if (std::getenv("RMPS_LA_DEBUG"))
            std::fprintf(stderr, "[probe] step=%.3g frame_e=%.4g prev=%.4g\n",
                         worst_step, frame_e, prev_probe_energy);
        prev_probe_energy = frame_e;
        return h;
    }

    void recompute_frame_timing() {
        if (state.mpeg.frac_slots_per_frame != 0.0) {
            if (state.mpeg.slot_lag <= (state.mpeg.frac_slots_per_frame - 1.0))
                state.mpeg.padding = 1;
            else
                state.mpeg.padding = 0;
            state.mpeg.slot_lag += (state.mpeg.padding - state.mpeg.frac_slots_per_frame);
        }
        state.mpeg.bits_per_frame =
            8 * (state.mpeg.whole_slots_per_frame + state.mpeg.padding);
        state.mean_bits =
            (state.mpeg.bits_per_frame - state.sideinfo_len) / state.mpeg.granules_per_frame;
    }

    /// RMS of the frame just analysed, in dBFS. Measured on `state.buffer`,
    /// i.e. the PCM this frame is built from — not on `staged`, which is the
    /// frame after it.
    std::optional<double> frame_level_db{};

    void run_frame(int stride) {
        // 1. PCM-domain transient analysis FIRST. This is bitrate-
        //    independent and produces the flags the ABR hint needs.
        bool any_transient = false;
        bool any_afs       = false;
        bool all_silent    = true;
        // Capture the previous frame's last-granule attack info before
        // analyse_frame overwrites the flag arrays (see EncoderState).
        {
            const int lastg = state.mpeg.granules_per_frame - 1;
            state.prev_frame_attack =
                state.transient_flags[lastg]
                || state.attack_from_silence_flags[lastg];
            state.prev_frame_attack_sub = state.attack_sub_index[lastg];
        }
        if (state.buffer[0]) {
            state.transient_gate.analyse_frame(
                state.buffer[0],
                samples_per_pass(),
                state.wave.channels,
                state.mpeg.granules_per_frame);
            state.frame_transient_granules = 0;
            state.frame_min_transient_cutoff_hz = 22050.0;
            state.prev_frame_last_sub_energy =
                state.sub_window_energies[state.mpeg.granules_per_frame - 1][2];
            for (int gr = 0; gr < state.mpeg.granules_per_frame; ++gr) {
                const auto d = state.transient_gate.decision_for(gr);
                state.transient_flags[gr]    = d.transient;
                state.pre_window_flags[gr]   = d.pre_window;
                state.attack_from_silence_flags[gr] = d.attack_from_silence;
                state.silent_flags[gr]       = d.silent;
                state.hf_transient_flags[gr] = d.hf_transient;
                state.transient_cutoff_hz[gr] = d.cutoff_hz;
                state.sub_window_energies[gr] = d.sub_energies;
                state.attack_sub_index[gr]    = d.attack_sub_index;
                state.energy_contrast[gr]     = d.energy_contrast;
                if (d.transient) {
                    ++state.frame_transient_granules;
                    any_transient = true;
                }
                if (d.attack_from_silence) any_afs = true;
                if (!d.silent) all_silent = false;
                if (d.cutoff_hz < state.frame_min_transient_cutoff_hz)
                    state.frame_min_transient_cutoff_hz = d.cutoff_hz;
                // pe[] is cleared here and filled for real in iteration_loop,
                // once the masking thresholds it is derived from exist. It
                // used to be left at 0, which silently disabled every
                // demand-driven path in the bit reservoir.
                for (int ch = 0; ch < state.wave.channels; ++ch)
                    state.pe[ch][gr] = 0.0;
            }
            // Cross-frame pre_window: if gr=0 of the current frame is a
            // transient, the last granule of the *previous* frame was the
            // pre-echo zone. We can't re-encode it, but we inject a
            // synthetic low threshold into the masking history so the
            // post-mask decay carries a stricter threshold into gr=0.
            state.cross_frame_pre_window =
                state.transient_flags[0]
                && !state.transient_gate.prev_frame_last_decision().transient;
            state.cross_frame_energy_contrast =
                state.cross_frame_pre_window
                    ? state.energy_contrast[0] : 1.0;
        }

        frame_level_db.reset();
        if (state.buffer[0]) {
            const std::size_t n =
                samples_per_pass() * static_cast<std::size_t>(state.wave.channels);
            double acc = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double v = state.buffer[0][i];
                acc += v * v;
            }
            constexpr double FULL_SCALE_SQ = 32768.0 * 32768.0;
            const double ms = acc / static_cast<double>(n);
            if (ms > 0.0)
                frame_level_db = 10.0 * std::log10(ms / FULL_SCALE_SQ);
        }

        // 2. Block types and MDCT, BEFORE the rate is chosen.
        //
        // Both are bitrate-independent (block types come from the transient
        // gate, the MDCT from the polyphase filterbank), so running them first
        // costs nothing and buys the rate controller a look at the actual
        // spectrum it is about to pay for. The alternative — choosing the rate
        // from PCM-domain heuristics and last frame's SDR — is what left decay
        // tails starved.
        bend::begin_frame(state);
        control::decide_block_types(state);
        mdct::analyse(state, stride);

        if (hf_extender && state.buffer[0]) {
            const std::size_t n_samples = samples_per_pass();
            hf_extender->process_frame(
                state.buffer[0], n_samples, state.wave.channels,
                state.mpeg.granules_per_frame, state.mdct_freq_ext,
                state.mdct_freq);
            state.hf_cutoff_sfb = hf_extender->cutoff_sfb(state.mpeg.samplerate_index);
        }

        // 3. ABR picks the bitrate: feed-forward from this frame's perceptual
        //    entropy, then the transient hint, which still overrides.
        if (abr) {
            const double frame_seconds =
                static_cast<double>(samples_per_pass()) / state.wave.samplerate;
            abr->set_demand(control::frame_demand_bits(state), frame_seconds);
            abr->set_transient_hint(any_afs, any_transient, all_silent);
            control::switch_bitrate(state, abr->next_target_kbps());
            state.policy_kbps = std::max(
                static_cast<double>(state.mpeg.bitr), abr->operating_kbps());
        } else {
            state.policy_kbps = static_cast<double>(state.mpeg.bitr);
        }
        recompute_frame_timing();

        quant::iteration_loop(state);
        l3::format_bitstream(state);
        bend::end_frame(state);
    }

    void copy_output() {
        last_output.assign(state.bs.data(),
                           state.bs.data() + state.bs.byte_position());
        state.bs.consume_position();

        // Xing/Info tag bookkeeping: every produced frame is exactly
        // bits_per_frame long once emitted (reservoir emission only lags,
        // it never resizes), so the size list drives both the frame count
        // and the seek TOC. The placeholder leads the very first output.
        frame_sizes.push_back(
            static_cast<std::uint32_t>(state.mpeg.bits_per_frame / 8));
        if (!tag_prepended) {
            build_vbr_tag(/*final_pass=*/false);
            last_output.insert(last_output.begin(),
                               tag_frame.begin(), tag_frame.end());
            tag_prepended = true;
        }

        // Build FrameStats from state.
        last_stats = {};
        last_stats.granules         = state.frame_granule_stats;
        last_stats.total_bits       = state.frame_total_bits;
        last_stats.total_budget     = state.frame_total_budget;
        if (state.frame_granule_count > 0) {
            double r = 0.0, p = 0.0, fu = 0.0, bl = 0.0;
            for (const auto& g : state.frame_granule_stats) {
                r += g.raw_sdr_db;
                p += g.perc_sdr_db;
                fu += g.fullness_db;
                bl += g.bleedless_db;
            }
            const double n = state.frame_granule_count;

            // --- Frame SDR: energy-weighted, not a mean of decibels ---------
            //
            // Averaging per-granule dB gives every granule the same vote
            // regardless of how much signal it carries. A ratio of summed
            // powers is the aggregate that weights each granule by what it
            // actually contributes, and `iteration_loop` was already
            // accumulating exactly that into `frame_raw_signal_pow` /
            // `frame_raw_dist` — which nothing consumed.
            //
            // Validated against the harness' own `segSNR_db`, which is an
            // independent time-domain measurement of the decoded file
            // (energy-weighted / mean-of-dB / segSNR):
            //
            //   mixed_music  48k   30.3 / 20.6 / 31.6
            //   mixed_music 128k   48.2 / 42.7 / 48.6
            //   mixed_music 320k   65.5 / 61.1 / 63.3
            //
            // The mean-of-dB is 11 dB low at 48 kbps on the clip nearest to
            // program material, and the error shrinks as the rate rises — i.e.
            // it is largest exactly where the rate loops have to work. The
            // spread comes from granules whose own ratio is extreme in either
            // direction (a near-silent granule coded exactly, or one zeroed
            // outright) getting a full vote at a tiny share of the energy.
            //
            // On the synthetic clips the two aggregations agree to a dB or two
            // and both diverge from segSNR — expected, since an MDCT-domain
            // ratio cannot see pre-echo (`transient_train`) or inter-channel
            // error (`stereo_decorrelated`). This change does not claim to fix
            // that; it fixes the aggregation.
            //
            // It matters beyond reporting: `target_sdr_db` steers the ABR rate
            // against this number and `quality_sdr_db` (`-q`) holds it as a
            // setpoint, so a frame-content-dependent bias of 11 dB is a bias in
            // the rate. At CBR the bitstream is unchanged (verified
            // bit-identical), so the risk is confined to the adaptive modes.
            //
            // RMPS_SDR_MEANDB=1 restores the old aggregation for comparison.
            static const bool mean_db = [] {
                const char* e = std::getenv("RMPS_SDR_MEANDB");
                return e && std::atoi(e) != 0;
            }();
            // The same argument applies to Fullness / Bleedless, and there it
            // is not reporting-only: `fb_score_db` is what the ABR and `-q`
            // loops actually hold as a setpoint (see the `fb_target` default
            // below), so a mean of per-granule dB was the aggregation the rate
            // was being steered by. Both are signal-to-failure ratios in the
            // same units as SDR, so the same fix applies: one ratio of summed
            // A-weighted powers over the frame.
            if (mean_db) {
                last_stats.mean_raw_sdr_db  = r / n;
                last_stats.mean_perc_sdr_db = p / n;
                last_stats.mean_fullness_db  = fu / n;
                last_stats.mean_bleedless_db = bl / n;
            } else {
                const double wsig = std::max(state.frame_w_signal_pow, 1.0);
                last_stats.mean_fullness_db =
                    10.0 * std::log10(wsig / std::max(state.frame_w_def, 1.0));
                last_stats.mean_bleedless_db =
                    10.0 * std::log10(wsig / std::max(state.frame_w_exc, 1.0));
                last_stats.mean_raw_sdr_db =
                    (state.frame_raw_signal_pow > 1.0 && state.frame_raw_dist > 1.0)
                        ? 10.0 * std::log10(state.frame_raw_signal_pow /
                                            state.frame_raw_dist)
                        : 0.0;
                last_stats.mean_perc_sdr_db =
                    (state.frame_w_signal_pow > 1.0 && state.frame_w_dist > 1.0)
                        ? 10.0 * std::log10(state.frame_w_signal_pow /
                                            state.frame_w_dist)
                        : 0.0;
            }
            // The worse of the two — now that both are signal-to-failure
            // ratios, "worse" is the smaller. Being full must not excuse
            // bleeding.
            last_stats.fb_score_db =
                std::min(last_stats.mean_fullness_db,
                         last_stats.mean_bleedless_db);
        }
        last_stats.hf_rdo_bits_saved = state.frame_hf_rdo_saved;
        last_stats.lpf_side_hz = post::lowpass_side_cutoff_hz(state.mpeg.bitr);
        if (state.hf_adaptor) last_stats.hf = state.hf_adaptor->state();
        last_stats.transient_granules  = state.frame_transient_granules;
        last_stats.transient_cutoff_hz = state.frame_min_transient_cutoff_hz;
        for (int gr = 0; gr < state.mpeg.granules_per_frame; ++gr)
            for (int ch = 0; ch < state.wave.channels; ++ch) {
                const int bt = state.block_type[ch][gr];
                if (bt == BLOCK_SHORT) ++last_stats.short_block_granules;
                else if (bt == BLOCK_START || bt == BLOCK_STOP)
                    ++last_stats.transition_block_granules;
            }
        // Per-granule window trace (RMPS_BLOCK_TRACE=1): one line per coded
        // granule, so a time-domain artefact found in the decoded file can be
        // mapped back to the window that produced it. The granule index is the
        // stream's own, i.e. sample offset / 576, which is what an external
        // analysis of the decode computes.
        if (std::getenv("RMPS_BLOCK_TRACE")) {
            static long long g_idx = 0;
            for (int gr = 0; gr < state.mpeg.granules_per_frame; ++gr) {
                std::fprintf(stderr, "[gtrace] gr=%lld bt=%d,%d tr=%d afs=%d "
                                     "pw=%d\n",
                             g_idx, state.block_type[0][gr],
                             state.block_type[state.wave.channels - 1][gr],
                             static_cast<int>(state.transient_flags[gr]),
                             static_cast<int>(state.attack_from_silence_flags[gr]),
                             static_cast<int>(state.pre_window_flags[gr]));
                ++g_idx;
            }
        }
        if (std::getenv("RMPS_BLOCK_STATS")) {
            static long long n_gr = 0, n_short = 0, n_trans = 0, n_att = 0;
            for (int gr = 0; gr < state.mpeg.granules_per_frame; ++gr)
                for (int ch = 0; ch < state.wave.channels; ++ch) {
                    ++n_gr;
                    const int bt = state.block_type[ch][gr];
                    if (bt == BLOCK_SHORT) ++n_short;
                    else if (bt == BLOCK_START || bt == BLOCK_STOP) ++n_trans;
                }
            for (int gr = 0; gr < state.mpeg.granules_per_frame; ++gr)
                if (state.transient_flags[gr]) ++n_att;
            static long long n_hf = 0, n_both = 0, n_pw = 0;
            for (int gr = 0; gr < state.mpeg.granules_per_frame; ++gr) {
                if (state.hf_transient_flags[gr]) ++n_hf;
                if (state.transient_flags[gr] && state.hf_transient_flags[gr])
                    ++n_both;
                if (state.pre_window_flags[gr]) ++n_pw;
            }
            std::fprintf(stderr,
                         "[blk] gr=%lld short=%lld (%.2f%%) trans=%lld "
                         "transient=%lld hf=%lld both=%lld prewin=%lld\n",
                         n_gr, n_short,
                         100.0 * static_cast<double>(n_short) /
                             static_cast<double>(n_gr ? n_gr : 1),
                         n_trans, n_att, n_hf, n_both, n_pw);
        }
        last_stats.attack_from_silence_granules = 0;
        for (int gr = 0; gr < state.mpeg.granules_per_frame; ++gr)
            if (state.attack_from_silence_flags[gr])
                ++last_stats.attack_from_silence_granules;
        last_stats.mask_alpha    = state.mask_params.alpha;
        last_stats.mask_lambda_R = state.mask_params.lambda_R;

        // Bitstream view for analysis: what the decoder will read, for every
        // granule and channel, straight from the side info and scalefactors
        // that format_bitstream just wrote.
        last_stats.bitrate_kbps = state.mpeg.bitr;
        last_stats.samplerate = state.wave.samplerate;
        last_stats.ms_stereo = (state.mpeg.mode_ext & 2) != 0;
        last_stats.reservoir_bits = state.ResvSize;
        last_stats.reservoir_max_bits = state.ResvMax;
        last_stats.overflow_bits = state.frame_overflow_bits;
        last_stats.lowpass_mid_hz = state.lpf_mid_applied_hz;
        last_stats.side.clear();
        for (int gr = 0; gr < state.mpeg.granules_per_frame; ++gr)
            for (int ch = 0; ch < state.wave.channels; ++ch) {
                const auto& gi = state.side_info.gr[gr].ch[ch].tt;
                GranuleSide g;
                g.gr = gr;
                g.ch = ch;
                g.block_type = gi.window_switching_flag ? gi.block_type : BLOCK_LONG;
                g.global_gain = gi.global_gain;
                g.scalefac_scale = gi.scalefac_scale;
                g.preflag = gi.preflag;
                g.scalefac_compress = gi.scalefac_compress;
                g.part2_bits = gi.part2_length;
                g.part2_3_bits = gi.part2_3_length;
                g.big_values = gi.big_values;
                g.count1 = gi.count1;
                g.table_select = gi.table_select;
                g.region0_count = gi.region0_count;
                g.region1_count = gi.region1_count;
                g.count1_table = gi.count1table_select;
                g.subblock_gain = gi.subblock_gain;
                g.scfsi = state.side_info.scfsi[ch];
                for (int b = 0; b < 21; ++b)
                    g.sf_long[static_cast<std::size_t>(b)] = static_cast<std::int8_t>(
                        state.scalefactor.l[gr][ch][static_cast<std::size_t>(b)]);
                for (int w = 0; w < 3; ++w)
                    for (int b = 0; b < 12; ++b)
                        g.sf_short[static_cast<std::size_t>(w)][static_cast<std::size_t>(b)] =
                            static_cast<std::int8_t>(state.scalefactor.s[gr][ch][static_cast<std::size_t>(w)][static_cast<std::size_t>(b)]);
                last_stats.side.push_back(g);
            }

        // Per-window-type quality census (RMPS_WINDOW_CENSUS=1).
        //
        // Short blocks are asked for by the transient gate, but the START and
        // STOP windows that must bracket them are forced by the bitstream's
        // switching rules, and nothing here ever measured what they cost. The
        // window analysis says START spreads a coding error over the same 36
        // samples as LONG while having a transition window's poorer frequency
        // resolution, so it should code worse at equal bits — this reports
        // whether it actually does, and by how much.
        if (std::getenv("RMPS_WINDOW_CENSUS")) {
            static long long n[4] = {};
            static double sdr[4] = {}, full[4] = {}, bits[4] = {};
            for (const auto& g : state.frame_granule_stats) {
                const int bt = g.block_type & 3;
                ++n[bt];
                sdr[bt] += g.raw_sdr_db;
                full[bt] += g.fullness_db;
                bits[bt] += g.bits;
            }
            static long long tick = 0;
            if ((++tick % 400) == 0) {
                static const char* nm[4] = {"LONG", "START", "SHORT", "STOP"};
                std::fprintf(stderr, "[win] ");
                for (int b = 0; b < 4; ++b)
                    if (n[b])
                        std::fprintf(stderr,
                                     "%s n=%lld sdr=%.2f full=%.2f bits=%.0f | ",
                                     nm[b], n[b], sdr[b] / n[b],
                                     full[b] / n[b], bits[b] / n[b]);
                std::fprintf(stderr, "\n");
            }
        }

        // Line-level decision flicker (RMPS_FLICKER=1).
        //
        // Measured externally at 48 kbps: sustained partials at 3-6 kHz move
        // 3.4 dB frame to frame in the *source* and 10.2 dB in the decode, and
        // the frames where a partial vanishes are only 1.5 dB below its own
        // running median. At 96 kbps the decode moves 4.1 dB, i.e. it tracks
        // the source. So the dashed harmonics are this encoder re-deciding,
        // not the material.
        //
        // This attributes the re-deciding. For lines whose *source* magnitude
        // barely changed between consecutive granules, it counts how often the
        // quantized value crossed zero anyway, and reports global_gain's own
        // step alongside — a wobbling granule-wide step moves every line's
        // rounding boundary at once and would show up as flicker on lines the
        // encoder never individually reconsidered.
        if (std::getenv("RMPS_FLICKER")) {
            static long long steady = 0, flipped = 0, gg_n = 0;
            static double gg_step = 0.0;
            static std::array<std::array<int, GRANULE_SAMPLES>,
                              MAX_CHANNELS> prev_ix{};
            static std::array<std::array<std::int32_t, GRANULE_SAMPLES>,
                              MAX_CHANNELS> prev_xr{};
            static std::array<int, MAX_CHANNELS> prev_gg{};
            static long long pk_n = 0;
            static double pk_step = 0.0;
            static bool have_prev = false;
            for (int gr = 0; gr < state.mpeg.granules_per_frame; ++gr) {
                for (int ch = 0; ch < state.wave.channels; ++ch) {
                    const auto& ix = state.l3_enc[ch][gr];
                    const auto& xr = state.mdct_freq[ch][gr];
                    const int gg = state.side_info.gr[gr].ch[ch].tt.global_gain;
                    if (have_prev && state.block_type[ch][gr] == 0) {
                        gg_step += std::abs(gg - prev_gg[ch]);
                        ++gg_n;
                        for (int i = 78; i < 158; ++i) {   // 3-6 kHz
                            const double a = std::abs(
                                static_cast<double>(prev_xr[ch][i]));
                            const double b = std::abs(
                                static_cast<double>(xr[i]));
                            if (a < 1.0 || b < 1.0) continue;
                            // "steady" = within 1.5 dB, the level at which the
                            // external measurement says drops start happening
                            // The *source* MDCT's own frame-to-frame step on
                            // its local maxima. An STFT says these partials
                            // move 3.4 dB; the MDCT is not shift-invariant, so
                            // a steady sinusoid's line magnitude also moves
                            // with its phase inside the block, and that motion
                            // is in the encoder's input, not its decisions.
                            if (i > 0 && i + 1 < GRANULE_SAMPLES &&
                                b > std::abs(static_cast<double>(xr[i - 1])) &&
                                b > std::abs(static_cast<double>(xr[i + 1]))) {
                                pk_step += std::abs(20.0 * std::log10(b / a));
                                ++pk_n;
                            }
                            if (std::abs(20.0 * std::log10(b / a)) > 1.5)
                                continue;
                            ++steady;
                            if ((prev_ix[ch][i] == 0) != (ix[i] == 0))
                                ++flipped;
                        }
                    }
                    prev_ix[ch] = ix;
                    prev_xr[ch] = xr;
                    prev_gg[ch] = gg;
                }
                have_prev = true;
            }
            static long long tick = 0;
            if ((++tick % 400) == 0)
                std::fprintf(stderr,
                             "[flick] steady_lines=%lld flipped=%lld (%.1f%%) "
                             "gg_step=%.2f quarter-dB  src_peak_step=%.2f dB "
                             "(n=%lld)\n",
                             steady, flipped,
                             100.0 * static_cast<double>(flipped) /
                                 static_cast<double>(steady ? steady : 1),
                             gg_step / static_cast<double>(gg_n ? gg_n : 1),
                             pk_step / static_cast<double>(pk_n ? pk_n : 1),
                             pk_n);
        }

        if (std::getenv("RMPS_SDR_DUMP")) {
            static long long fno = 0;
            std::fprintf(stderr,
                         "[sdr] f=%lld raw=%.2f perc=%.2f bits=%d budget=%d "
                         "resv=%d\n",
                         fno++, last_stats.mean_raw_sdr_db,
                         last_stats.mean_perc_sdr_db, last_stats.total_bits,
                         last_stats.total_budget, state.ResvSize);
        }

        // Feed the closing-loop controllers.
        if (state.hf_adaptor) state.hf_adaptor->observe(last_stats);
        if (abr) {
            const double frame_seconds =
                static_cast<double>(samples_per_pass()) / state.wave.samplerate;
            std::optional<double> sdr;
            // Quality mode can hold the Fullness/Bleedless score instead of
            // SDR. SDR sums "did not keep" and "invented" into one number, so
            // it cannot distinguish a hole from a fabrication and rates both
            // the noise fill and the shaper's boost as fine; FB scores them
            // separately and reports the worse. The controller is unchanged —
            // both are dB, so the same loop holds either.
            // Default ON. RMPS_Q_SDR=1 restores the old SDR setpoint.
            static const bool fb_target = [] {
                const char* e = std::getenv("RMPS_Q_SDR");
                return !(e && std::atoi(e) != 0);
            }();
            if (state.frame_granule_count > 0)
                sdr = fb_target ? last_stats.fb_score_db
                    : std::getenv("RMPS_RAW_SDR")
                          ? last_stats.mean_raw_sdr_db
                          : last_stats.mean_perc_sdr_db;
            if (std::getenv("RMPS_FB_DUMP"))
                std::fprintf(stderr, "[fb] full=%.2f bleed=%.2f score=%.2f "
                                     "perc_sdr=%.2f\n",
                             last_stats.mean_fullness_db,
                             last_stats.mean_bleedless_db,
                             last_stats.fb_score_db,
                             last_stats.mean_perc_sdr_db);
            // Feed the PRODUCED frame size, not the emitted byte count:
            // reservoir emission is lagged (a banking frame emits nothing,
            // a later one emits several), which would whipsaw the PID.
            abr->observe(state.mpeg.bits_per_frame, frame_seconds, sdr,
                         frame_level_db, state.frame_overflow_bits);
        }
        if (std::getenv("RMPS_OVF_DEBUG")) {
            static long long ovf = 0, total = 0, frames = 0, ovf_frames = 0;
            ovf += state.frame_overflow_bits;
            total += state.mpeg.bits_per_frame;
            ++frames;
            if (state.frame_overflow_bits > 0) ++ovf_frames;
            std::fprintf(stderr,
                         "[ovf] frames=%lld overflowing=%lld (%.1f%%) "
                         "wasted=%lld bits (%.2f%% of stream) kbps=%d\n",
                         frames, ovf_frames,
                         100.0 * static_cast<double>(ovf_frames) /
                             static_cast<double>(frames),
                         ovf,
                         100.0 * static_cast<double>(ovf) /
                             static_cast<double>(total ? total : 1),
                         state.mpeg.bitr);
        }
    }
};

Encoder::Encoder() : impl_(std::make_unique<Impl>()) {}
Encoder::~Encoder() = default;

BandLayout band_layout(int samplerate) {
    BandLayout out;
    const int idx = find_samplerate_index(samplerate);
    if (idx < 0) return out;
    const auto& l = tables::scale_fact_band_index[static_cast<std::size_t>(idx)];
    const auto& sh = tables::scale_fact_band_index_short[static_cast<std::size_t>(idx)];
    out.long_edges.assign(l.begin(), l.end());
    out.short_edges.assign(sh.begin(), sh.end());
    return out;
}

std::unique_ptr<Encoder> Encoder::create(const Config& cfg) {
    auto e = std::unique_ptr<Encoder>(new Encoder());
    if (!e->impl_->init(cfg)) return nullptr;
    return e;
}

std::size_t Encoder::samples_per_pass() const noexcept {
    return impl_->samples_per_pass();
}

int Encoder::coded_samplerate() const noexcept {
    return impl_->state.wave.samplerate;
}

std::span<const std::uint8_t> Encoder::encode(PcmChunk pcm) {
    impl_->out_accum.clear();

    // Unbuffered: the chunk *is* the frame, exactly as before.
    if (!impl_->buffered()) {
        impl_->encode_one(pcm.data, pcm.num_samples);
        return std::span<const std::uint8_t>(impl_->out_accum.data(),
                                             impl_->out_accum.size());
    }

    // Buffered: the zero-phase lowpass releases samples in blocks once its
    // backward pass has enough future context, and the resampler's output does
    // not map to whole coded frames either (1152 samples at 32 kHz is 1587.6 at
    // 44.1), so PCM is collected and drained a frame at a time. A call can
    // therefore produce zero frames or several — already legal, since the
    // one-frame lookahead makes the first call return nothing and reservoir
    // emission is lagged.
    if (impl_->pre_lpf) {
        const std::size_t ch = static_cast<std::size_t>(impl_->state.wave.channels);
        impl_->lpf_out.clear();
        impl_->pre_lpf->process(pcm.data, pcm.num_samples / ch, impl_->lpf_out);
        impl_->to_framing(impl_->lpf_out.data(), impl_->lpf_out.size());
    } else {
        impl_->to_framing(pcm.data, pcm.num_samples);
    }
    impl_->drain_frames();
    return std::span<const std::uint8_t>(impl_->out_accum.data(),
                                         impl_->out_accum.size());
}

std::span<const std::uint8_t> Encoder::flush() {
    auto& st = impl_->state;

    // Drain the resampler's filter history and code whatever whole frames come
    // out of it, zero-padding the last partial frame. Skipping this drops the
    // last ~32 input samples plus anything short of a frame still buffered.
    std::vector<std::uint8_t> resampled_tail;
    if (impl_->buffered()) {
        impl_->out_accum.clear();
        // Release the lowpass's held tail (exactly the input's length), then
        // the resampler's filter history.
        if (impl_->pre_lpf) {
            impl_->lpf_out.clear();
            impl_->pre_lpf->flush(impl_->lpf_out);
            impl_->to_framing(impl_->lpf_out.data(), impl_->lpf_out.size());
        }
        if (impl_->resampler && !impl_->resampler->passthrough())
            impl_->resampler->flush(impl_->rs_pending);
        const std::size_t ch = static_cast<std::size_t>(st.wave.channels);
        const std::size_t need = impl_->samples_per_pass() * ch;
        if (!impl_->rs_pending.empty() && impl_->rs_pending.size() % need)
            impl_->rs_pending.resize(
                (impl_->rs_pending.size() / need + 1) * need, 0);
        for (std::size_t pos = 0; pos + need <= impl_->rs_pending.size();
             pos += need)
            impl_->encode_one(impl_->rs_pending.data() + pos, need);
        impl_->rs_pending.clear();
        resampled_tail = impl_->out_accum;
    }

    // The staged frame has no successor, so it encodes with no lookahead.
    std::vector<std::uint8_t> tail;
    if (impl_->staged_valid) {
        st.buffer[0]     = impl_->staged.data();
        st.buffer_idx[0] = 0;
        if (st.wave.channels == 2) {
            st.buffer[1]     = impl_->staged.data();
            st.buffer_idx[1] = 1;
        }
        st.lookahead_valid  = false;
        st.lookahead_attack_gr0 = false;
        st.lookahead_attack = false;
        st.lookahead_demand = 0.0;
        impl_->staged_valid = false;
        impl_->run_frame(st.wave.channels);
        impl_->copy_output();
        tail = impl_->last_output;
    }

    // Emit frames still waiting on future main data (reservoir tail).
    l3::flush_bitstream(impl_->state);
        subband::fx_probe_report();
    impl_->last_output.assign(
        impl_->state.bs.data(),
        impl_->state.bs.data() + impl_->state.bs.byte_position());
    impl_->state.bs.consume_position();
    if (!tail.empty())
        impl_->last_output.insert(impl_->last_output.begin(),
                                  tail.begin(), tail.end());
    if (!resampled_tail.empty())
        impl_->last_output.insert(impl_->last_output.begin(),
                                  resampled_tail.begin(), resampled_tail.end());
    return std::span<const std::uint8_t>(impl_->last_output.data(),
                                         impl_->last_output.size());
}

std::span<const std::uint8_t> Encoder::vbr_tag() {
    if (impl_->frame_sizes.empty()) return {};
    impl_->build_vbr_tag(/*final_pass=*/true);
    return std::span<const std::uint8_t>(impl_->tag_frame.data(),
                                         impl_->tag_frame.size());
}

const FrameStats& Encoder::last_frame_stats() const noexcept {
    return impl_->last_stats;
}

const AbrState* Encoder::abr_state() const noexcept {
    if (!impl_->abr) return nullptr;
    static thread_local AbrState st;
    st = impl_->abr->state();
    return &st;
}

EncoderBends& Encoder::bends() noexcept { return impl_->state.bends; }

const BendView& Encoder::bend_view() const noexcept { return impl_->state.bend.view; }

void Encoder::set_target_kbps(int kbps) {
    if (impl_->abr) {
        // For ABR we mutate the controller state's target by reconstructing.
        // (Re-creating the controller resets its PID memory; that's intended.)
        auto o = control::AbrController::Options{};
        o.target_kbps   = kbps;
        o.target_sdr_db = impl_->cfg.target_sdr_db;
        if (impl_->cfg.abr_kp) o.kp = *impl_->cfg.abr_kp;
        if (impl_->cfg.abr_ki) o.ki = *impl_->cfg.abr_ki;
        if (impl_->cfg.abr_kd) o.kd = *impl_->cfg.abr_kd;
        o.min_kbps      = impl_->cfg.abr_min_kbps;
        o.max_kbps      = impl_->cfg.abr_max_kbps;
        o.mpeg_version  = impl_->state.mpeg.version;
        impl_->abr = std::make_unique<control::AbrController>(o);
    } else {
        control::switch_bitrate(impl_->state, kbps);
    }
}

} // namespace rmps
