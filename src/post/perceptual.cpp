#include "post/perceptual.hpp"
#include "core/sync.hpp"
#include "core/tables.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace rmps::post {

namespace {

constexpr double MS_IMAGE_FACTOR = 0.82;   // 500..3 kHz preserved (was 0.70)
constexpr double MS_OUTER_FACTOR = 0.58;   // outside that band (was 0.45)

double a_weight_db(double f) {
    if (f < 10.0) return -50.0;
    const double f2  = f * f;
    const double num = 12194.0 * 12194.0 * f2 * f2;
    const double den = (f2 + 20.6 * 20.6) * (f2 + 12194.0 * 12194.0)
                     * std::sqrt((f2 + 107.7 * 107.7) * (f2 + 737.9 * 737.9));
    if (den <= 0.0) return -50.0;
    return 20.0 * std::log10(std::max(num / den, 1e-6)) + 2.0;
}

struct CacheKey { int sr_index; bool side; bool operator==(const CacheKey& o) const noexcept { return sr_index == o.sr_index && side == o.side; }};
struct CacheHash { std::size_t operator()(const CacheKey& k) const noexcept {
    return static_cast<std::size_t>(k.sr_index) * 2u + (k.side ? 1u : 0u);
}};

std::unordered_map<CacheKey, std::array<double, NUM_SFB_LONG_>, CacheHash> g_cache;
sync::Mutex g_cache_mu;

} // namespace

const std::array<double, NUM_SFB_LONG_>&
perceptual_weights(int samplerate_index, int samplerate, bool is_side_channel) {
    CacheKey key {samplerate_index, is_side_channel};
    sync::Lock lk(g_cache_mu);
    auto it = g_cache.find(key);
    if (it != g_cache.end()) return it->second;

    std::array<double, NUM_SFB_LONG_> w {};
    const auto& sfb = tables::scale_fact_band_index[samplerate_index];
    for (int s = 0; s < NUM_SFB_LONG_; ++s) {
        const double center_sample = (sfb[s] + sfb[s + 1]) * 0.5;
        const double f_hz = center_sample * samplerate / (2.0 * 576.0);
        double v = std::pow(10.0, a_weight_db(f_hz) / 20.0);
        if (is_side_channel) {
            v *= (f_hz >= 500.0 && f_hz <= 3000.0) ? MS_IMAGE_FACTOR : MS_OUTER_FACTOR;
        }
        w[s] = v;
    }
    double wmax = *std::max_element(w.begin(), w.end());
    if (wmax <= 0.0) wmax = 1.0;
    for (auto& x : w) x = std::max(x / wmax, 0.05);
    auto [ins, _] = g_cache.emplace(key, w);
    return ins->second;
}

} // namespace rmps::post
