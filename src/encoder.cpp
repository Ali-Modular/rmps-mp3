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

int granules_per_frame_for(int version) {
    return version == 3 ? 2 : 1;
}

} // namespace

struct Encoder::Impl {
    EncoderState               state;
    std::vector<std::uint8_t>  last_output;
    FrameStats                 last_stats;
    std::unique_ptr<control::AbrController> abr;
    Config                     cfg;

    bool init(const Config& c) {
        cfg = c;
        const int initial_kbps = c.abr_target_kbps.value_or(c.bitrate);

        const int sr_idx = find_samplerate_index(c.samplerate);
        if (sr_idx < 0) return false;
        const int ver    = mpeg_version_for(sr_idx);
        const int br_idx = find_bitrate_index(initial_kbps, ver);
        if (br_idx < 0) return false;

        state.wave.channels   = c.channels;
        state.wave.samplerate = c.samplerate;
        state.mpeg.bitr       = initial_kbps;
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

        if (c.abr_target_kbps) {
            control::AbrController::Options o;
            o.target_kbps = *c.abr_target_kbps;
            if (c.target_sdr_db) o.target_sdr_db = c.target_sdr_db;
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
        if (abr) {
            control::switch_bitrate(state, abr->next_target_kbps());
        }
        recompute_frame_timing();
        mdct::analyse(state, stride);
        quant::iteration_loop(state);
        l3::format_bitstream(state);
    }

    void copy_output() {
        last_output.assign(state.bs.data(),
                           state.bs.data() + state.bs.byte_position());
        state.bs.consume_position();

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

        // Feed the closing-loop controllers.
        if (state.hf_adaptor) state.hf_adaptor->observe(last_stats);
        if (abr) {
            const double frame_seconds =
                static_cast<double>(samples_per_pass()) / state.wave.samplerate;
            std::optional<double> sdr;
            if (state.frame_granule_count > 0) sdr = last_stats.mean_perc_sdr_db;
            abr->observe(static_cast<int>(last_output.size() * 8u),
                         frame_seconds, sdr);
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
    impl_->last_output.clear();
    return std::span<const std::uint8_t>(impl_->last_output.data(), 0u);
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
        o.min_kbps      = impl_->cfg.abr_min_kbps;
        o.max_kbps      = impl_->cfg.abr_max_kbps;
        o.mpeg_version  = impl_->state.mpeg.version;
        impl_->abr = std::make_unique<control::AbrController>(o);
    } else {
        control::switch_bitrate(impl_->state, kbps);
    }
}

} // namespace rmps
