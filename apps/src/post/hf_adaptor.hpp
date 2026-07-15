#pragma once
#include "rmps/stats.hpp"

namespace rmps {
struct EncoderState;
struct FrameStats;
namespace post {

class HFAdaptor {
public:
  explicit HFAdaptor(double shelf_hz = 8000.0, double max_atten_m_db = 1.0,
                     double max_atten_s_db = 1.0, double target_sdr_db = 25.0,
                     double ema_alpha = 0.45);

  /// Detect transient + apply shelf to s.mdct_freq.
  void apply(EncoderState &s);

  /// Feed back last frame's stats to bias next frame's attenuation.
  void observe(const FrameStats &fs);

  HFAdaptorState state() const noexcept;

private:
  double shelf_hz_;
  double max_atten_m_db_;
  double max_atten_s_db_;
  double target_sdr_db_;
  double ema_alpha_;
  double cur_atten_db_ = 0.0;
  double prev_energy_ = 0.0;
  bool transient_ = false;
  double last_total_atten_db_ = 0.0;
};

} // namespace post
} // namespace rmps
