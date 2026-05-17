#include "quant/iteration_loop.hpp"
#include "quant/quantize.hpp"
#include "quant/beam_search.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"
#include "core/reservoir.hpp"
#include "post/ms_stereo.hpp"
#include "post/lowpass.hpp"
#include "post/hf_adaptor.hpp"
#include "post/perceptual.hpp"

#include <cmath>
#include <cstdlib>

namespace rmps::quant {

void iteration_loop(EncoderState& s) {
    // ----- Per-frame pre-quantization processing -----
    post::apply_ms_stereo(s);
    s.hf_adaptor_apply(); // <- call wrapper if defined; otherwise direct.
    post::apply_lowpass(s);

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
