#include "post/ms_stereo.hpp"
#include "core/encoder_state.hpp"
#include "core/fixed.hpp"

#include <algorithm>

namespace rmps::post {

namespace { constexpr double INV_SQRT2 = 0.7071067811865475; }

void apply_ms_stereo(EncoderState& s) {
    if (s.wave.channels != 2) return;
    for (int gr = 0; gr < s.mpeg.granules_per_frame; ++gr) {
        auto& L = s.mdct_freq[0][gr];
        auto& R = s.mdct_freq[1][gr];
        for (int i = 0; i < GRANULE_SAMPLES; ++i) {
            const std::int64_t l = L[i], r = R[i];
            const double m = (l + r) * INV_SQRT2;
            const double t = (l - r) * INV_SQRT2;
            L[i] = fx::i32(static_cast<std::int64_t>(m));
            R[i] = fx::i32(static_cast<std::int64_t>(t));
        }
    }
}

} // namespace rmps::post
