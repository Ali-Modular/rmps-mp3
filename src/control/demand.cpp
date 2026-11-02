#include "control/demand.hpp"
#include "core/encoder_state.hpp"
#include "core/sync.hpp"
#include "core/tables.hpp"
#include "post/perceptual.hpp"
#include "post/psy_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>

namespace rmps::control {

namespace {

namespace psy = post::psy;

// The absolute threshold of hearing is applied here at full strength, and
// relieving it does nothing at all: dropping it by 30 dB left `pe` identical
// to the last digit on real material. Spread masking dominates the threshold
// everywhere the signal exists, so the ATH floor almost never binds, and the
// intuition that "the ATH is what makes a quiet frame cheap" is simply wrong.
//
// The deeper consequence is worth stating, because it is not obvious: since
// `thr` is built from the band energies themselves, e/thr is scale-invariant,
// so `pe` measures spectral *shape* — complexity, tonality, contrast — and
// carries no information about level at all. A quiet passage reads cheap only
// when it is genuinely simpler, never merely because it is quiet.

/// Whether to divide the spread threshold by the total spreading gain.
///
/// The long model normalizes, so that a threshold is a weighted *average* of
/// masker powers rather than a raw sum — without it, centre bands are inflated
/// relative to edge bands purely because they have more neighbours. That is
/// the right choice for a coding threshold. It is not obviously the right one
/// for a demand estimate: normalizing makes `thr` track `e` closely, which is
/// what flattens `pe`.
bool normalize_spread() {
    if (const char* e = std::getenv("RMPS_DEMAND_NORM"))
        return std::atoi(e) != 0;
    return true;
}

/// Per-SFB geometry. Same construction as the long masking model's, so the
/// `pe` produced here lands on the same scale as `EncoderState::pe`.
struct Geom {
    std::array<double, NUM_SFB_LONG_> center_hz{};
    std::array<double, NUM_SFB_LONG_> center_bark{};
    std::array<double, NUM_SFB_LONG_> width_bark{};
    std::array<double, NUM_SFB_LONG_> ath_pow{};
    std::array<int, NUM_SFB_LONG_>    n_bins{};
    std::array<int, NUM_SFB_LONG_ + 1> edge{};
};

Geom build_geom(int samplerate_index, int samplerate) {
    Geom g{};
    const auto& sfb = tables::scale_fact_band_index[samplerate_index];
    for (int b = 0; b < NUM_SFB_LONG_; ++b) {
        const double lo_hz = sfb[b] * samplerate / (2.0 * 576.0);
        const double hi_hz = sfb[b + 1] * samplerate / (2.0 * 576.0);
        const double c_hz  = 0.5 * (lo_hz + hi_hz);
        g.center_hz[b]   = c_hz;
        g.center_bark[b] = psy::bark_of_hz(c_hz);
        g.width_bark[b]  =
            std::max(psy::bark_of_hz(hi_hz) - psy::bark_of_hz(lo_hz), 0.05);
        g.n_bins[b] = sfb[b + 1] - sfb[b];
        g.edge[b]   = sfb[b];
        g.ath_pow[b] = psy::spl_to_power(psy::ath_min_db_over(lo_hz, hi_hz));
    }
    g.edge[NUM_SFB_LONG_] = sfb[NUM_SFB_LONG_];
    return g;
}

const Geom& geom_for(int samplerate_index, int samplerate) {
    static std::array<Geom, tables::NUM_SAMPLERATES> cache{};
    static std::array<bool, tables::NUM_SAMPLERATES> ready{};
    static sync::Mutex mu;
    sync::Lock lk(mu);
    const auto idx = static_cast<std::size_t>(samplerate_index);
    if (!ready[idx]) {
        cache[idx] = build_geom(samplerate_index, samplerate);
        ready[idx] = true;
    }
    return cache[idx];
}

/// Spreading gain in dB from masker to target, in the Bark domain. Same
/// asymmetric, level-dependent skirt the long model uses.
double spread_db(double dz_bark, double f_masker_khz, double masker_spl_db) {
    if (dz_bark <= 0.0)
        return -psy::LOWER_SLOPE_DB_PER_BARK * (-dz_bark);
    return -psy::upper_slope_db_per_bark(f_masker_khz, masker_spl_db) * dz_bark;
}

/// `pe` for one granule's spectrum.
double granule_demand(const std::int32_t* xr, const Geom& g) {
    std::array<double, NUM_SFB_LONG_> e_band{};
    std::array<double, NUM_SFB_LONG_> masker_power{};
    std::array<double, NUM_SFB_LONG_> masker_spl{};

    for (int b = 0; b < NUM_SFB_LONG_; ++b) {
        double e = 0.0;
        for (int i = g.edge[b]; i < g.edge[b + 1]; ++i) {
            const double v = xr[i];
            e += v * v;
        }
        e_band[b] = e;
        const double ton =
            psy::tonality_from_sfm(xr + g.edge[b], g.n_bins[b]);
        masker_power[b] = e * std::pow(10.0, psy::smr_db(ton) / 10.0);
        masker_spl[b] =
            psy::effective_masker_spl(e, g.width_bark[b], ton);
    }

    std::array<double, NUM_SFB_LONG_> thr{};
    for (int b = 0; b < NUM_SFB_LONG_; ++b) {
        double t = 0.0, norm = 0.0;
        for (int m = 0; m < NUM_SFB_LONG_; ++m) {
            const double s_db =
                spread_db(g.center_bark[b] - g.center_bark[m],
                          g.center_hz[m] / 1000.0, masker_spl[m]);
            const double gain = std::pow(10.0, s_db / 10.0);
            t    += masker_power[m] * gain;
            norm += gain;
        }
        const double spread = (normalize_spread() && norm > 0.0) ? (t / norm) : t;
        thr[b] = std::max(spread, g.ath_pow[b]);
    }

    return psy::perceptual_entropy_of(e_band.data(), thr.data(),
                                      g.n_bins.data(), NUM_SFB_LONG_);
}

} // namespace

double frame_demand_bits(const EncoderState& s) {
    const auto& g = geom_for(s.mpeg.samplerate_index, s.wave.samplerate);
    double pe = 0.0;
    for (int ch = 0; ch < s.wave.channels; ++ch)
        for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr)
            pe += granule_demand(s.mdct_freq[ch][gr].data(), g);
    return pe;
}

} // namespace rmps::control
