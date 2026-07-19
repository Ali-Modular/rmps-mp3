#include "rmps/rmps.hpp"

#include "control/abr_pid.hpp"
#include "core/encoder_state.hpp"
#include "core/l3_bitstream.hpp"
#include "core/mdct.hpp"
#include "core/subband.hpp"
#include "core/tables.hpp"
#include "post/hf_adaptor.hpp"
#include "post/lowpass.hpp"
#include "quant/iteration_loop.hpp"
#include "quant/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>

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

struct Encoder::Impl {
    EncoderState               state;
    std::vector<std::uint8_t>  last_output;
    FrameStats                 last_stats;
    std::unique_ptr<control::AbrController> abr;

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
        const int initial_kbps = c.abr_target_kbps.value_or(c.bitrate);

        const int sr_idx = find_samplerate_index(c.samplerate);
        if (sr_idx < 0) return false;
        const int ver    = mpeg_version_for(sr_idx);
        
        int snapped_kbps = 0;
        int br_idx = -1;
        if (c.abr_target_kbps) {
            br_idx = find_nearest_bitrate_index(initial_kbps, ver, snapped_kbps);
        } else {
            br_idx = find_bitrate_index(initial_kbps, ver);
            snapped_kbps = initial_kbps;
        }
        if (br_idx < 0) return false;

        state.wave.channels   = c.channels;
        state.wave.samplerate = c.samplerate;
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

        state.cfg_mask_alpha    = c.mask_alpha;
        state.cfg_mask_lambda_R = c.mask_lambda_R;
        state.cfg_lf_strict_db        = c.lf_strict_db;
        state.cfg_lf_strict_cutoff_hz = c.lf_strict_cutoff_hz;
        state.pre_gain          = std::pow(10.0, c.pre_gain_db / 20.0);
        state.hf_weight_boost   = std::max(1.0, c.hf_weight_boost);
        state.cfg_hf_zero_boost_db = std::clamp(c.hf_zero_boost_db, 0.0, 24.0);
        state.cfg_noise_shaping_alpha = std::clamp(c.noise_shaping_alpha, 0.0, 0.95);
        state.cfg_waveform_reg  = c.waveform_reg;
        state.cfg_adaptive_ms   = c.adaptive_joint_stereo;
        state.ms_prev_frame     = true;
        state.cfg_block_switching = c.block_switching;
        state.prev_block_type   = {BLOCK_LONG, BLOCK_LONG};
        state.transient_gate.set_hf_transient_threshold(c.short_block_hf_threshold);

        // Cross-frame bit reservoir (main_data_begin). ResvSize/ResvMax plan
        // the banking (reservoir::adjust / frame_end keep everything above
        // ResvMax stuffed and the bank byte-aligned); l3_bitstream carries
        // the authoritative BYTE accounting at emission. The cap must stay
        // under the main_data_begin field range (511 bytes MPEG-1, 255
        // MPEG-2) with margin.
        state.ResvMax  = (ver == 3) ? 8 * 440 : 8 * 200;
        if (std::getenv("RMPS_NO_RESV")) state.ResvMax = 0;  // diagnostic
        state.ResvSize = 0;
        // Reset the cross-frame reservoir/FIFO state for this stream.
        state.md_fifo.clear();
        state.md_emitted = 0;
        state.resv_bytes = 0;
        state.pending_frames.clear();

        if (c.abr_target_kbps) {
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
                // pe[] stays at default 0: bit reservoir is disabled
                // (see encoder.cpp init). The bitrate index itself now
                // carries the transient signal via the ABR hint below.
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

        // 2. ABR picks the bitrate, biased by the transient hint.
        if (abr) {
            abr->set_transient_hint(any_afs, any_transient, all_silent);
            control::switch_bitrate(state, abr->next_target_kbps());
        }
        recompute_frame_timing();

        // 3. Decide per-(ch, gr) block types from the transient flags before
        //    MDCT, so analyse() can pick the matching analysis window.
        control::decide_block_types(state);

        mdct::analyse(state, stride);
        quant::iteration_loop(state);
        l3::format_bitstream(state);
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
            double r = 0.0, p = 0.0;
            for (const auto& g : state.frame_granule_stats) {
                r += g.raw_sdr_db;
                p += g.perc_sdr_db;
            }
            last_stats.mean_raw_sdr_db  = r / state.frame_granule_count;
            last_stats.mean_perc_sdr_db = p / state.frame_granule_count;
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
        last_stats.attack_from_silence_granules = 0;
        for (int gr = 0; gr < state.mpeg.granules_per_frame; ++gr)
            if (state.attack_from_silence_flags[gr])
                ++last_stats.attack_from_silence_granules;
        last_stats.mask_alpha    = state.mask_params.alpha;
        last_stats.mask_lambda_R = state.mask_params.lambda_R;

        // Feed the closing-loop controllers.
        if (state.hf_adaptor) state.hf_adaptor->observe(last_stats);
        if (abr) {
            const double frame_seconds =
                static_cast<double>(samples_per_pass()) / state.wave.samplerate;
            std::optional<double> sdr;
            if (state.frame_granule_count > 0) sdr = last_stats.mean_perc_sdr_db;
            // Feed the PRODUCED frame size, not the emitted byte count:
            // reservoir emission is lagged (a banking frame emits nothing,
            // a later one emits several), which would whipsaw the PID.
            abr->observe(state.mpeg.bits_per_frame, frame_seconds, sdr);
        }
    }
};

Encoder::Encoder() : impl_(std::make_unique<Impl>()) {}
Encoder::~Encoder() = default;

std::unique_ptr<Encoder> Encoder::create(const Config& cfg) {
    auto e = std::unique_ptr<Encoder>(new Encoder());
    if (!e->impl_->init(cfg)) return nullptr;
    return e;
}

std::size_t Encoder::samples_per_pass() const noexcept {
    return impl_->samples_per_pass();
}

std::span<const std::uint8_t> Encoder::encode(PcmChunk pcm) {
    impl_->state.buffer[0]     = pcm.data;
    impl_->state.buffer_idx[0] = 0;
    if (impl_->state.wave.channels == 2) {
        impl_->state.buffer[1]     = pcm.data;
        impl_->state.buffer_idx[1] = 1;
    }
    impl_->run_frame(impl_->state.wave.channels);
    impl_->copy_output();
    return std::span<const std::uint8_t>(impl_->last_output.data(),
                                         impl_->last_output.size());
}

std::span<const std::uint8_t> Encoder::flush() {
    // Emit frames still waiting on future main data (reservoir tail).
    l3::flush_bitstream(impl_->state);
    impl_->last_output.assign(
        impl_->state.bs.data(),
        impl_->state.bs.data() + impl_->state.bs.byte_position());
    impl_->state.bs.consume_position();
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
