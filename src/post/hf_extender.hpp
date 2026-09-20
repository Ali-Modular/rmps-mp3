#pragma once

#include "core/encoder_state.hpp"
#include <array>
#include <memory>

namespace rmps::post {

struct HFExtenderConfig {
  double cutoff_hz = -1.0;   ///< < 0: auto-detect
  double max_out_hz = 20000.0;  ///< Capped at 20000 Hz by default
  double obj_gain_db = 0.0;
  double res_gain_db = -1.0;
  double tilt_db_oct = 0.0;
  double struct_amt = 1.0;
  double env_innov = 1.0;
  int env_order = 0;
  double env_rho = 0.998;
  int cep_q = 30;
  int grid = 256;
  bool inharm = true;
  bool hpattern = true;
  double pat_min_frac = 0.35;
  int look = 2;
};

class HFExtender {
public:
  HFExtender();
  ~HFExtender();

  void init(const HFExtenderConfig &cfg, double sample_rate, int channels);

  /// Process the current PCM frame and generate synthesized high-frequency
  /// MDCT spectra in `out_mdct_ext[ch][gr][576]` for bins where high frequency
  /// is missing (cutoff to Nyquist).
  /// `ref_mdct` is the encoder's original MDCT frequency array (used for
  /// reference energy scaling).
  void process_frame(
      const std::int16_t *pcm_interleaved, size_t num_samples, int channels,
      int granules_per_frame,
      std::array<std::array<std::array<std::int32_t, GRANULE_SAMPLES>,
                            MAX_GRANULES>,
                 MAX_CHANNELS> &out_mdct_ext,
      const std::array<std::array<std::array<std::int32_t, GRANULE_SAMPLES>,
                                  MAX_GRANULES>,
                       MAX_CHANNELS> &ref_mdct);

  void process_frame(
      const double *pcm_interleaved, size_t num_samples, int channels,
      int granules_per_frame,
      std::array<std::array<std::array<std::int32_t, GRANULE_SAMPLES>,
                            MAX_GRANULES>,
                 MAX_CHANNELS> &out_mdct_ext,
      const std::array<std::array<std::array<std::int32_t, GRANULE_SAMPLES>,
                                  MAX_GRANULES>,
                       MAX_CHANNELS> &ref_mdct);

  double detected_cutoff_hz() const;
  int cutoff_sfb(int samplerate_index) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rmps::post
