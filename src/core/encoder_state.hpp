#pragma once
#include "core/bitstream.hpp"
#include "core/side_info.hpp"
#include "post/hf_adaptor.hpp"
#include "rmps/config.hpp"
#include "rmps/stats.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace rmps {

/// Mirrors Encoder's EncoderMpeg.
struct MpegState {
  int bitr = 128; ///< kbps
  Emphasis emph = Emphasis::None;
  bool copyright = false;
  bool original = true;

  int version = 3; ///< MPEG_I=3 / II=2 / 2.5=0
  int layer = 1;   ///< Layer III bitstream value
  int granules_per_frame = 2;
  int padding = 0;
  int bits_per_frame = 0;
  int bits_per_slot = 8;
  double frac_slots_per_frame = 0.0;
  double slot_lag = 0.0;
  int whole_slots_per_frame = 0;
  int bitrate_index = 0;
  int samplerate_index = 0;
  bool crc = false;
  int ext = 0;
  int mode_ext = 0;
  Mode mode = Mode::JointStereo;
};

struct WaveState {
  int channels = 2;
  int samplerate = 44100;
};

/// Full encoder state. Single owner: Encoder::Impl.
struct EncoderState {
  WaveState wave;
  MpegState mpeg;

  Bitstream bs{4096};
  SideInfo side_info{};
  int sideinfo_len = 0;
  int mean_bits = 0;
  Scalefac scalefactor{};

  // PCM input pointer (one per logical channel; same pointer with offset
  // 0/1 for interleaved inputs).
  std::array<const std::int16_t *, MAX_CHANNELS> buffer{nullptr, nullptr};
  std::array<std::size_t, MAX_CHANNELS> buffer_idx{0, 0};

  // Per-granule perceptual entropy (placeholder; reservoir uses it but the
  // simple encoder leaves it 0).
  std::array<std::array<double, MAX_GRANULES>, MAX_CHANNELS> pe{};

  // Quantized indices. [ch][gr][576]
  std::array<std::array<std::array<int, GRANULE_SAMPLES>, MAX_GRANULES>,
             MAX_CHANNELS>
      l3_enc{};

  // Subband sample cache. [ch][gr_plus_1][k][band]
  std::array<
      std::array<std::array<std::array<std::int32_t, SUBBANDS>, K_PER_GRANULE>,
                 3>,
      MAX_CHANNELS>
      l3_sb_sample{};

  // MDCT freq coefficients. [ch][gr][576]
  std::array<
      std::array<std::array<std::int32_t, GRANULE_SAMPLES>, MAX_GRANULES>,
      MAX_CHANNELS>
      mdct_freq{};

  int ResvSize = 0;
  int ResvMax = 0;

  // ----- loop tables (initialised once) -----
  std::array<double, 128> steptab{};
  std::array<std::int32_t, 128> steptabi{};
  std::array<int, 10000> int2idx{};

  // ----- subband state -----
  std::array<int, MAX_CHANNELS> subband_off{0, 0};
  std::array<std::array<std::int32_t, HAN_SIZE>, MAX_CHANNELS> subband_x{};
  std::array<std::array<std::int32_t, 64>, SUBBANDS> subband_fl{};

  // ----- MDCT tables -----
  std::array<std::array<std::int32_t, 36>, K_PER_GRANULE> mdct_cos_l{};

  // ----- per-iteration scratch (refreshed per granule) -----
  const std::int32_t *xr = nullptr;
  std::array<std::int32_t, GRANULE_SAMPLES> xrsq{};
  std::array<std::int32_t, GRANULE_SAMPLES> xrabs{};
  std::int32_t xrmax = 0;

  // ----- Post-processing modules -----
  std::unique_ptr<post::HFAdaptor> hf_adaptor;
  void hf_adaptor_apply();

  // ----- Per-frame aggregate stats (populated by iteration_loop) -----
  std::vector<GranuleStats> frame_granule_stats;
  double frame_raw_signal_pow = 0.0;
  double frame_raw_dist = 0.0;
  double frame_w_signal_pow = 0.0;
  double frame_w_dist = 0.0;
  int frame_granule_count = 0;
  int frame_total_bits = 0;
  int frame_total_budget = 0;
  int frame_hf_rdo_saved = 0;
};

} // namespace rmps
