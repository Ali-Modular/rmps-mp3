#include "quant/iteration_loop.hpp"
#include "quant/quantize.hpp"
#include "quant/beam_search.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include "core/reservoir.hpp"
#include "post/ms_stereo.hpp"
#include "post/lowpass.hpp"
#include "post/hf_adaptor.hpp"
#include "post/masking.hpp"
#include "post/perceptual.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace rmps::quant {

namespace {

#if 0  // outer_loop_sfs — see fallback note below; kept for reference.

int quantize_granule_sfs(int* ix, int stepsize, int gr, int ch, EncoderState& s) {
    auto& gi = s.side_info.gr[gr].ch[ch].tt;
    const auto& sfb_table = tables::scale_fact_band_index[s.mpeg.samplerate_index];
    const std::int32_t scalei_global = s.steptabi[stepsize + 127];
    const double       step_global   = s.steptab[stepsize + 127];

    int max_val = 0;
    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
        const int sf = s.scalefactor.l[gr][ch][sfb];
        const int eff_sf = sf + (gi.preflag ? tables::pretab[sfb] : 0);
        const int shift  = eff_sf * (1 + gi.scalefac_scale);
        // Guard against shift-overflow (shift > 30 makes 1<<shift UB).
        if (shift < 0 || shift > 30) return 16384;
        const std::int64_t scalei =
            static_cast<std::int64_t>(scalei_global) << shift;
        if (scalei <= 0 || scalei > 0x7fffffffLL) return 16384;
        const double step_band = step_global * static_cast<double>(1 << shift);

        const int start = sfb_table[sfb];
        const int end   = sfb_table[sfb + 1];
        for (int i = start; i < end; ++i) {
            const std::int32_t x = s.xrabs[i];
            const std::int64_t prod =
                static_cast<std::int64_t>(x) * scalei
              + static_cast<std::int64_t>(0x80000000LL);
            const std::int32_t ln = fx::i32(prod >> 32);
            int v;
            if (ln >= 0 && ln < 10000) {
                v = s.int2idx[static_cast<std::size_t>(ln)];
            } else {
                const double dbl =
                    static_cast<double>(x) * step_band * 4.656612875e-10;
                v = static_cast<int>(std::pow(std::max(dbl, 0.0), 0.75));
            }
            if (v > 8191) v = 8191;
            ix[i] = v;
            if (v > max_val) max_val = v;
        }
    }
    // Region beyond sfb 20 (16+ kHz at 44.1k) is encoded as zero by the
    // beam search; match that here for bitstream consistency.
    for (int i = sfb_table[NUM_SFB_LONG]; i < GRANULE_SAMPLES; ++i) ix[i] = 0;
    return max_val;
}

/// Bin-search the global gain to find the finest step that fits the
/// rate budget, using SF-aware quantization. Mirrors the legacy
/// bin_search_StepSize but calls our quantize_granule_sfs.
int bin_search_StepSize_sfs(int desired_rate, int* ix, int gr, int ch,
                            EncoderState& s) {
    auto& gi = s.side_info.gr[gr].ch[ch].tt;
    int next_val = -120, count = 120;
    while (count > 1) {
        const int half = count / 2;
        int bit;
        if (quantize_granule_sfs(ix, next_val + half, gr, ch, s) > 8192) {
            bit = 100000;
        } else {
            calc_runlen(ix, gi);
            bit  = count1_bitcount(ix, gi);
            subdivide(gi, s);
            bigv_tab_select(ix, gi);
            bit += bigv_bitcount(ix, gi);
        }
        if (bit < desired_rate) count = half;
        else { next_val += half; count -= half; }
    }
    return next_val;
}

/// Coarsen stepsize until the bit count fits `max_bits`. Mirrors the
/// legacy inner_loop but calls quantize_granule_sfs.
int inner_loop_sfs(int* ix, int max_bits, int gr, int ch, EncoderState& s) {
    auto& gi = s.side_info.gr[gr].ch[ch].tt;
    if (max_bits < 0) gi.quantizerStepSize -= 1;
    int bits = 0;
    while (true) {
        gi.quantizerStepSize += 1;
        // Safety: cap stepsize so we don't run away.
        if (gi.quantizerStepSize > 119) { gi.quantizerStepSize = 119; }
        while (quantize_granule_sfs(ix, gi.quantizerStepSize, gr, ch, s) > 8192) {
            gi.quantizerStepSize += 1;
            if (gi.quantizerStepSize > 119) { gi.quantizerStepSize = 119; break; }
        }
        calc_runlen(ix, gi);
        bits  = count1_bitcount(ix, gi);
        subdivide(gi, s);
        bigv_tab_select(ix, gi);
        bits += bigv_bitcount(ix, gi);
        if (bits <= max_bits) break;
        if (gi.quantizerStepSize >= 119) break;  // can't coarsen further
    }
    return bits;
}

/// SF-aware outer_loop. Caller populates s.scalefactor.l + gi
/// (inherited from last_good_granule), and we solve for global gain.
int outer_loop_sfs(int max_bits, int* ix, int gr, int ch, EncoderState& s) {
    auto& gi = s.side_info.gr[gr].ch[ch].tt;
    gi.quantizerStepSize = bin_search_StepSize_sfs(max_bits, ix, gr, ch, s);
    gi.part2_length      = part2_length(gr, ch, s);
    const int huff_bits  = max_bits - gi.part2_length;
    const int bits       = inner_loop_sfs(ix, huff_bits, gr, ch, s);
    gi.part2_3_length    = gi.part2_length + bits;
    return gi.part2_3_length;
}

#endif  // outer_loop_sfs block

/// Zero MDCT coefficients above `cutoff_hz` for `(ch, gr)`. Applied per-granule
/// on top of the channel-wide LPF when the transient gate fires.
void apply_transient_cutoff(EncoderState& s, int ch, int gr, double cutoff_hz) {
    const double nyq = s.wave.samplerate * 0.5;
    if (cutoff_hz >= nyq - 1.0) return;
    int cutoff_bin = static_cast<int>(cutoff_hz * 576.0 * 2.0 / s.wave.samplerate);
    if (cutoff_bin < 0) cutoff_bin = 0;
    if (cutoff_bin >= GRANULE_SAMPLES) return;
    auto& arr = s.mdct_freq[ch][gr];
    for (int i = cutoff_bin; i < GRANULE_SAMPLES; ++i) arr[i] = 0;
}

} // namespace

void iteration_loop(EncoderState& s) {
    // ----- Per-frame pre-quantization processing -----
    post::apply_ms_stereo(s);
    s.hf_adaptor_apply();
    post::apply_lowpass(s);

    // Per-granule transient cutoff (kicks pre-echo mitigation): applied to
    // every channel of any granule the gate flagged.
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
        const double cutoff = s.transient_cutoff_hz[gr];
        if (cutoff < s.wave.samplerate * 0.5 - 1.0) {
            for (int ch = 0; ch < s.wave.channels; ++ch) {
                // M channel (ch 0) keeps a slightly wider band than S so the
                // attack's body still has spectral content; S is killed more
                // aggressively (stereo side is least audible during a strike).
                const double ch_cutoff = (ch == 0 && s.wave.channels == 2)
                    ? std::min(cutoff * 1.4, 22050.0)
                    : cutoff;
                apply_transient_cutoff(s, ch, gr, ch_cutoff);
            }
        }
    }

    // Refresh masking model parameters and compute per-(ch, gr) thresholds
    // BEFORE beam_search sees this granule. Note: history advances in
    // (gr, ch) order so post-mask is causal frame-by-frame.
    s.mask_params = post::make_masking_params(s);
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
        for (int ch = 0; ch < s.wave.channels; ++ch) {
            post::compute_masking_thresholds(
                s, ch, gr, s.mask_params,
                s.mask_history[ch],
                s.mask_thresholds[ch][gr]);
        }
    }

    int gr0_sf_per_ch[MAX_CHANNELS][NUM_SFB_LONG];
    bool gr0_sf_valid[MAX_CHANNELS] = {false, false};

    // Aggregate stats for FrameStats.
    double raw_signal_total = 0.0;
    double raw_dist_total   = 0.0;
    double w_signal_total   = 0.0;
    double w_dist_total     = 0.0;
    int    granule_count    = 0;
    int    granule_bits_sum = 0;
    int    granule_budget_sum = 0;
    int    hf_rdo_total = 0;
    s.frame_outer_loop_fallbacks = 0;
    s.frame_granule_stats.clear();

    for (int ch = 0; ch < s.wave.channels; ++ch) {
        for (auto& v : s.side_info.scfsi[ch]) v = 0;
        for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
            auto* ix = s.l3_enc[ch][gr].data();
            s.xr = s.mdct_freq[ch][gr].data();

            s.xrsq.fill(0);
            s.xrabs.fill(0);
            s.xrmax = 0;
            for (int i = 0; i < GRANULE_SAMPLES; ++i) {
                const std::int32_t v = s.xr[i];
                const std::int64_t sq = static_cast<std::int64_t>(v) * v
                                      + static_cast<std::int64_t>(0x40000000LL);
                s.xrsq[i] = fx::i32(sq >> 31);
                const std::int32_t a = std::abs(v);
                s.xrabs[i] = a;
                if (a > s.xrmax) s.xrmax = a;
            }

            auto& gi = s.side_info.gr[gr].ch[ch].tt;
            gi.sfb_lmax = NUM_SFB_LONG;
            // ABR carries the silence/transient signal via the per-frame
            // bitrate index now — mean_bits is already small for silent
            // frames and large for attack frames. No further per-granule
            // budget shaping needed here.
            const int max_bits = reservoir::max_reservoir_bits(s.pe[ch][gr], s);

            for (auto& v : s.scalefactor.l[gr][ch]) v = 0;
            for (auto& v : gi.slen) v = 0;
            gi.part2_3_length      = 0;
            gi.big_values          = 0;
            gi.count1              = 0;
            gi.scalefac_compress   = 0;
            gi.table_select        = {0, 0, 0};
            gi.region0_count       = 0;
            gi.region1_count       = 0;
            gi.part2_length        = 0;
            gi.preflag             = 0;
            gi.scalefac_scale      = 0;
            gi.count1table_select  = 0;

            if (s.xrmax > 0) {
                const int* gr0_ptr = (gr == 1 && gr0_sf_valid[ch])
                                   ? gr0_sf_per_ch[ch] : nullptr;
                BeamResult br = beam_search(max_bits, ix, gr, ch, s, gr0_ptr);
                if (!br.ok) {
                    ++s.frame_outer_loop_fallbacks;
                    for (auto& v : s.side_info.scfsi[ch]) v = 0;
                    gi.scalefac_compress = 0;
                    gi.scalefac_scale    = 0;
                    gi.preflag           = 0;
                    for (int sfb = 0; sfb < NUM_SFB_LONG + 1; ++sfb)
                        s.scalefactor.l[gr][ch][sfb] = 0;
                    gi.part2_3_length = outer_loop(max_bits, ix, gr, ch, s);
                } else {
                    granule_bits_sum   += gi.part2_3_length;
                    granule_budget_sum += max_bits;
                    raw_signal_total   += br.raw_signal_pow;
                    raw_dist_total     += br.raw_dist;
                    w_signal_total     += br.weighted_signal_pow;
                    w_dist_total       += br.cum_dist;
                    hf_rdo_total       += br.hf_rdo_bits_saved;
                    ++granule_count;

                    GranuleStats gs {};
                    gs.gr = gr; gs.ch = ch;
                    gs.gain     = gi.quantizerStepSize;
                    gs.preflag  = gi.preflag;
                    gs.scale    = gi.scalefac_scale;
                    gs.compress = gi.scalefac_compress;
                    for (int b = 0; b < 4; ++b) gs.scfsi_pat[b] = s.side_info.scfsi[ch][b];
                    gs.bits   = gi.part2_3_length;
                    gs.budget = max_bits;
                    if (br.raw_signal_pow > 1.0 && br.raw_dist > 1.0)
                        gs.raw_sdr_db = 10.0 * std::log10(br.raw_signal_pow / br.raw_dist);
                    if (br.weighted_signal_pow > 1.0 && br.cum_dist > 1.0)
                        gs.perc_sdr_db = 10.0 * std::log10(br.weighted_signal_pow / br.cum_dist);
                    gs.hf_rdo_bits_saved = br.hf_rdo_bits_saved;
                    s.frame_granule_stats.push_back(gs);
                }
                if (gr == 0) {
                    for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb)
                        gr0_sf_per_ch[ch][sfb] = s.scalefactor.l[0][ch][sfb];
                    gr0_sf_valid[ch] = true;
                }
            }
            reservoir::adjust(gi, s);
            gi.global_gain = gi.quantizerStepSize + 210;
        }
    }

    // Publish frame totals on s.
    s.frame_raw_signal_pow = raw_signal_total;
    s.frame_raw_dist       = raw_dist_total;
    s.frame_w_signal_pow   = w_signal_total;
    s.frame_w_dist         = w_dist_total;
    s.frame_granule_count  = granule_count;
    s.frame_total_bits     = granule_bits_sum;
    s.frame_total_budget   = granule_budget_sum;
    s.frame_hf_rdo_saved   = hf_rdo_total;

    reservoir::frame_end(s);
}

} // namespace rmps::quant
