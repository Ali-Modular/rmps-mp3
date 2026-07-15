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
#include <cstring>
#include <memory>
#include <optional>

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

struct FrameBackupState {
  std::array<std::array<int, MAX_GRANULES>, MAX_CHANNELS> block_type;
  std::array<int, MAX_CHANNELS> prev_block_type;
  std::array<std::array<std::array<int, GRANULE_SAMPLES>, MAX_GRANULES>, MAX_CHANNELS> l3_enc;
  std::array<std::array<std::array<std::int32_t, GRANULE_SAMPLES>, MAX_GRANULES>, MAX_CHANNELS> mdct_freq;
  SideInfo side_info;
  Scalefac scalefactor;
  std::array<post::MaskingHistory, MAX_CHANNELS> mask_history;
  std::vector<GranuleStats> frame_granule_stats;
  double frame_raw_signal_pow;
  double frame_raw_dist;
  double frame_bass_dist;
  double frame_w_signal_pow;
  double frame_w_dist;
  int frame_granule_count;
  int frame_total_bits;
  int frame_total_budget;
  int frame_hf_rdo_saved;
  int frame_outer_loop_fallbacks;
  // Analysis-side state consumed/advanced by mdct::analyse and
  // iteration_loop. mdct::analyse reads PCM through buffer_idx and advances
  // the polyphase filterbank history (subband_x/subband_off) plus the MDCT
  // overlap cache (l3_sb_sample); iteration_loop advances the adaptive-MS
  // hysteresis, the HF adaptor EMA, the reservoir and last_good_granule.
  // The A/B block-switching compare runs the whole chain twice per frame,
  // so all of it must be part of the snapshot.
  std::array<std::size_t, MAX_CHANNELS> buffer_idx;
  std::array<int, MAX_CHANNELS> subband_off;
  std::array<std::array<std::int32_t, HAN_SIZE>, MAX_CHANNELS> subband_x;
  std::array<
      std::array<std::array<std::array<std::int32_t, SUBBANDS>, K_PER_GRANULE>,
                 3>,
      MAX_CHANNELS>
      l3_sb_sample;
  bool ms_prev_frame;
  int mode_ext;
  int resv_size;
  std::array<EncoderState::LastGoodGranule, MAX_CHANNELS> last_good_granule;
  std::optional<post::HFAdaptor> hf_adaptor;
};

FrameBackupState backup_frame_state(const EncoderState& s) {
  FrameBackupState b{};
  b.block_type = s.block_type;
  b.prev_block_type = s.prev_block_type;
  b.l3_enc = s.l3_enc;
  b.mdct_freq = s.mdct_freq;
  b.side_info = s.side_info;
  b.scalefactor = s.scalefactor;
  b.mask_history = s.mask_history;
  b.frame_granule_stats = s.frame_granule_stats;
  b.frame_raw_signal_pow = s.frame_raw_signal_pow;
  b.frame_raw_dist = s.frame_raw_dist;
  b.frame_bass_dist = s.frame_bass_dist;
  b.frame_w_signal_pow = s.frame_w_signal_pow;
  b.frame_w_dist = s.frame_w_dist;
  b.frame_granule_count = s.frame_granule_count;
  b.frame_total_bits = s.frame_total_bits;
  b.frame_total_budget = s.frame_total_budget;
  b.frame_hf_rdo_saved = s.frame_hf_rdo_saved;
  b.frame_outer_loop_fallbacks = s.frame_outer_loop_fallbacks;
#ifndef RMPS_SIMULATE_OLD_AB
  b.buffer_idx = s.buffer_idx;
  b.subband_off = s.subband_off;
  b.subband_x = s.subband_x;
  b.l3_sb_sample = s.l3_sb_sample;
  b.ms_prev_frame = s.ms_prev_frame;
  b.mode_ext = s.mpeg.mode_ext;
  b.resv_size = s.ResvSize;
  b.last_good_granule = s.last_good_granule;
  if (s.hf_adaptor) b.hf_adaptor = *s.hf_adaptor;
#endif
  return b;
}

void restore_frame_state(EncoderState& s, const FrameBackupState& b) {
  s.block_type = b.block_type;
  s.prev_block_type = b.prev_block_type;
  s.l3_enc = b.l3_enc;
  s.mdct_freq = b.mdct_freq;
  s.side_info = b.side_info;
  s.scalefactor = b.scalefactor;
  s.mask_history = b.mask_history;
  s.frame_granule_stats = b.frame_granule_stats;
  s.frame_raw_signal_pow = b.frame_raw_signal_pow;
  s.frame_raw_dist = b.frame_raw_dist;
  s.frame_bass_dist = b.frame_bass_dist;
  s.frame_w_signal_pow = b.frame_w_signal_pow;
  s.frame_w_dist = b.frame_w_dist;
  s.frame_granule_count = b.frame_granule_count;
  s.frame_total_bits = b.frame_total_bits;
  s.frame_total_budget = b.frame_total_budget;
  s.frame_hf_rdo_saved = b.frame_hf_rdo_saved;
  s.frame_outer_loop_fallbacks = b.frame_outer_loop_fallbacks;
#ifndef RMPS_SIMULATE_OLD_AB
  s.buffer_idx = b.buffer_idx;
  s.subband_off = b.subband_off;
  s.subband_x = b.subband_x;
  s.l3_sb_sample = b.l3_sb_sample;
  s.ms_prev_frame = b.ms_prev_frame;
  s.mpeg.mode_ext = b.mode_ext;
  s.ResvSize = b.resv_size;
  s.last_good_granule = b.last_good_granule;
  if (b.hf_adaptor && s.hf_adaptor) *s.hf_adaptor = *b.hf_adaptor;
#endif
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
        state.cfg_hf_gain_boost_14k = std::max(1.0, c.hf_gain_boost_14k);
        state.cfg_hf_zero_boost_db = std::clamp(c.hf_zero_boost_db, 0.0, 24.0);
        state.cfg_noise_shaping_alpha = std::clamp(c.noise_shaping_alpha, 0.0, 0.95);
        state.cfg_waveform_reg  = c.waveform_reg;

        // Derive beam search knobs from effort level [0..10] if not explicitly overridden
        const int effort = std::clamp(c.effort, 0, 10);
        static constexpr int EFFORT_BEAM_WIDTH[]    = { 1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 48 };
        static constexpr int EFFORT_GAIN_WINDOW[]   = { 0, 1, 2, 3, 4,  6,  8, 12, 16, 20, 24 };
        static constexpr int EFFORT_SF_CANDIDATE[]  = { 0, 1, 2, 3, 4,  4,  6,  8, 10, 12, 15 };

        const int bw = (c.beam_width >= 0) ? c.beam_width : EFFORT_BEAM_WIDTH[effort];
        const int gw = (c.gain_window >= 0) ? c.gain_window : EFFORT_GAIN_WINDOW[effort];
        const int sfc = (c.sf_max_candidate >= 0) ? c.sf_max_candidate : EFFORT_SF_CANDIDATE[effort];

        state.cfg_beam_width         = std::max(1, bw);
        state.cfg_gain_window        = std::max(0, gw);
        state.cfg_sf_max_candidate   = std::clamp(sfc, 0, 15);
        state.cfg_adaptive_ms   = c.adaptive_joint_stereo;
        state.ms_prev_frame     = true;
        state.cfg_block_switching = c.block_switching;
        state.prev_block_type   = {BLOCK_LONG, BLOCK_LONG};
        state.transient_gate.set_hf_transient_threshold(c.short_block_hf_threshold);

        // Cross-frame bit reservoir. ResvMax is recomputed per frame in
        // recompute_frame_timing() (it depends on the current bitrate);
        // main_data_begin itself is derived from the md_fifo byte counts in
        // l3_bitstream, which is the drift-free source of truth.
        state.ResvMax  = 0;
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

        // Reservoir ceiling for the current bitrate: bounded by the
        // main_data_begin field width (9 bits -> 511 bytes for MPEG-1,
        // 8 bits -> 255 bytes for MPEG-2) and by the decoder's main_data
        // buffer (7680 bits for MPEG-1), which must hold the reservoir
        // plus one frame's own main data.
        const int avail = state.mpeg.bits_per_frame - state.sideinfo_len;
        const int field_cap = (state.mpeg.version == 3) ? 511 * 8 : 255 * 8;
        const int buf_cap   = ((state.mpeg.version == 3) ? 7680 : 4096) - avail;
        state.ResvMax = std::clamp(buf_cap, 0, field_cap);
    }

    void run_frame(int stride) {
        // 1. PCM-domain transient analysis FIRST. This is bitrate-
        //    independent and produces the flags the ABR hint needs.
        bool any_transient = false;
        bool any_afs       = false;
        bool all_silent    = true;
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

        // 2.5 Reservoir demand hints. max_reservoir_bits() lends a granule up
        // to 60% of the reservoir when pe*3.1 exceeds its flat share, and
        // trims the grant (donation) when it falls clearly below — the beam
        // search spends whatever it is given, so steady granules must under-
        // request slightly for the reservoir to ever fill. Attacks borrow,
        // steady content donates ~15%, silence donates the maximum.
        {
            const double share =
                static_cast<double>(state.mean_bits) / state.wave.channels;
            for (int gr = 0; gr < state.mpeg.granules_per_frame; ++gr) {
                double f = 0.85;
                if (state.attack_from_silence_flags[gr])   f = 2.6;
                else if (state.transient_flags[gr])        f = 2.2;
                else if (state.silent_flags[gr])           f = 0.0;
                for (int ch = 0; ch < state.wave.channels; ++ch)
                    state.pe[ch][gr] = share * f / 3.1;
            }
        }

        // 3. Decide per-(ch, gr) block types from the transient flags before
        //    MDCT, so analyse() can pick the matching analysis window.
        control::decide_block_types(state);

        bool has_switching = false;
        if (state.cfg_block_switching) {
            for (int ch = 0; ch < state.wave.channels; ++ch) {
                for (int gr = 0; gr < state.mpeg.granules_per_frame; ++gr) {
                    if (state.block_type[ch][gr] != BLOCK_LONG) {
                        has_switching = true;
                        break;
                    }
                }
                if (has_switching) break;
            }
        }

        if (has_switching) {
            // Snapshot the FULL pre-analysis state. Both paths run
            // mdct::analyse, which consumes PCM (buffer_idx) and advances the
            // polyphase filterbank history (subband_x/off) and the MDCT
            // overlap cache (l3_sb_sample). Without restoring these between
            // the runs, Path B reads PCM past the end of the frame and the
            // filterbank history advances twice, desyncing the MDCT overlap
            // with the next frame — decoded as a low-frequency thump right
            // after every transient.
            const auto pre_state = backup_frame_state(state);

            // Path A: With Block Switching (Transient Gate's Decision)
            mdct::analyse(state, stride);
            quant::iteration_loop(state);
            // Penalise short-block bass damage: raw_dist captures unweighted
            // quantization noise across all bands. Short blocks that damage
            // sustained bass produce disproportionately high raw_dist; adding
            // a fraction of it biases the A/B compare toward long blocks when
            // bass fidelity is at stake (the "boff" avoidance lever).
            const double cost_A = state.frame_w_dist;
            const double bass_dist_A = state.frame_bass_dist;
            const auto state_A = backup_frame_state(state);

            // Restore pre-analysis state for Path B
            restore_frame_state(state, pre_state);
            // Force long-only blocks — but the previous frame is already
            // emitted, so the window sequence it left behind must still be
            // honoured: after a SHORT granule only SHORT or STOP may follow.
            // decide_block_types encodes that entry constraint in its gr=0
            // choice (SHORT/STOP there means the previous frame ended SHORT),
            // so keep a STOP window at gr=0 in that case; STOP is a long-
            // layout block, so Path B still contains no short windows.
            // Blindly forcing LONG here emitted SHORT->LONG transitions whose
            // uncancelled overlap aliasing decodes as a bass thump.
            const int entry_bt0 = state.block_type[0][0];
            const bool need_stop =
                (entry_bt0 == BLOCK_SHORT || entry_bt0 == BLOCK_STOP);
            for (int ch = 0; ch < MAX_CHANNELS; ++ch) {
                for (int gr = 0; gr < MAX_GRANULES; ++gr) {
                    state.block_type[ch][gr] = BLOCK_LONG;
                }
                if (need_stop) state.block_type[ch][0] = BLOCK_STOP;
            }

            // Path B: Forced Long-Only Blocks
            mdct::analyse(state, stride);
            quant::iteration_loop(state);
            const double cost_B = state.frame_w_dist;
            const double bass_dist_B = state.frame_bass_dist;

            // Compare costs and choose the better path. Path B is already
            // the live state; only Path A needs a restore.
            // If Path A's bass distortion is worse than Path B's by more than 10%,
            // force Path B (Long Block) to protect the bass.
            bool force_long = (bass_dist_A > bass_dist_B * 1.10);

            if (cost_B < cost_A || force_long) {
                for (int ch = 0; ch < state.wave.channels; ++ch) {
                    state.prev_block_type[ch] = BLOCK_LONG;
                }
            } else {
                restore_frame_state(state, state_A);
            }
        } else {
            mdct::analyse(state, stride);
            quant::iteration_loop(state);
        }
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
            // With the reservoir on, frames are emitted with a small lag
            // (a frame leaves the queue once its main_data slot fills), so
            // last_output.size() no longer tracks per-frame cost. The frame's
            // header-declared size is the true rate spent this pass.
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
    auto& st = impl_->state;
    l3::flush_bitstream(st);
    impl_->last_output.assign(st.bs.data(), st.bs.data() + st.bs.byte_position());
    st.bs.consume_position();
    return std::span<const std::uint8_t>(impl_->last_output.data(),
                                         impl_->last_output.size());
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
