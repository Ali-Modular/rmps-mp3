#include "post/hf_extender.hpp"
#include "core/fixed.hpp"
#include "core/tables.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace rmps::post {

namespace {

using cd = std::complex<double>;
static const double PI = 3.14159265358979323846;

inline double clampd(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

inline double wrapPi(double x) {
  while (x > PI)
    x -= 2.0 * PI;
  while (x < -PI)
    x += 2.0 * PI;
  return x;
}

inline double dbToLin(double db) { return std::pow(10.0, db / 20.0); }

// ================================== FFT =====================================

struct FFT {
  int n = 0;
  std::vector<int> rev;
  std::vector<cd> tw;

  void init(int N) {
    n = N;
    rev.assign(n, 0);
    int lg = 0;
    while ((1 << lg) < n)
      ++lg;
    for (int i = 0; i < n; ++i) {
      int r = 0;
      for (int b = 0; b < lg; ++b)
        if (i & (1 << b))
          r |= 1 << (lg - 1 - b);
      rev[i] = r;
    }
    tw.assign(n / 2, cd(0, 0));
    for (int i = 0; i < n / 2; ++i)
      tw[i] = std::polar(1.0, -2.0 * PI * i / n);
  }

  void run(std::vector<cd> &a, bool inverse) const {
    for (int i = 0; i < n; ++i)
      if (i < rev[i])
        std::swap(a[i], a[rev[i]]);
    for (int len = 2; len <= n; len <<= 1) {
      int step = n / len;
      for (int i = 0; i < n; i += len) {
        for (int j = 0; j < len / 2; ++j) {
          cd w = tw[j * step];
          if (inverse)
            w = std::conj(w);
          cd u = a[i + j];
          cd v = a[i + j + len / 2] * w;
          a[i + j] = u + v;
          a[i + j + len / 2] = u - v;
        }
      }
    }
  }
};

// ================================== MCLT ====================================

struct MCLT {
  int M = 0, N = 0;
  FFT fft;
  std::vector<double> win;
  std::vector<cd> preA;
  std::vector<cd> postA;
  std::vector<cd> buf;

  void init(int M_) {
    M = M_;
    N = 2 * M;
    fft.init(N);
    win.resize(N);
    for (int n = 0; n < N; ++n)
      win[n] = std::sin(PI * (n + 0.5) / N);
    preA.resize(N);
    for (int n = 0; n < N; ++n)
      preA[n] = std::polar(1.0, -PI * n / N);
    postA.resize(M);
    for (int k = 0; k < M; ++k)
      postA[k] = std::polar(1.0, -PI * (k + 0.5) * (M + 1) / (2.0 * M));
    buf.resize(N);
  }

  void analyze(const double *x, std::vector<cd> &X) {
    for (int n = 0; n < N; ++n)
      buf[n] = preA[n] * (x[n] * win[n]);
    fft.run(buf, false);
    X.resize(M);
    for (int k = 0; k < M; ++k)
      X[k] = postA[k] * buf[k];
  }

  void synthesize(const std::vector<double> &C, std::vector<double> &y) {
    for (int k = 0; k < M; ++k)
      buf[k] = C[k] * std::conj(postA[k]);
    for (int k = M; k < N; ++k)
      buf[k] = cd(0, 0);
    fft.run(buf, true);
    y.resize(N);
    double s = 2.0 / M;
    for (int n = 0; n < N; ++n)
      y[n] = s * (std::conj(preA[n]) * buf[n]).real() * win[n];
  }

  double binToHz(double bin, double sr) const { return (bin + 0.5) * sr / N; }
  double hzToBin(double hz, double sr) const { return hz * N / sr - 0.5; }
};

// =========================== sinusoidal atom kernel =========================

struct Kernel {
  static constexpr int R = 5;
  static constexpr int W = 2 * R + 1;
  int D = 129;
  std::vector<cd> Kc, Ks;

  void build(MCLT &t) {
    int M = t.M, N = t.N;
    int k0 = M / 2;
    Kc.assign(D * W, cd(0, 0));
    Ks.assign(D * W, cd(0, 0));
    std::vector<double> x(N);
    std::vector<cd> X;
    for (int i = 0; i < D; ++i) {
      double d = double(i) / (D - 1);
      double nu = (k0 + d + 0.5) / N;
      for (int n = 0; n < N; ++n)
        x[n] = std::cos(2 * PI * nu * n);
      t.analyze(x.data(), X);
      for (int j = -R; j <= R; ++j)
        Kc[i * W + (j + R)] = X[k0 + j];
      for (int n = 0; n < N; ++n)
        x[n] = std::sin(2 * PI * nu * n);
      t.analyze(x.data(), X);
      for (int j = -R; j <= R; ++j)
        Ks[i * W + (j + R)] = X[k0 + j];
    }
  }

  void get(double d, cd *kc, cd *ks) const {
    double f = clampd(d, 0.0, 1.0) * (D - 1);
    int i0 = (int)f;
    if (i0 > D - 2)
      i0 = D - 2;
    double fr = f - i0;
    for (int j = 0; j < W; ++j) {
      kc[j] = Kc[i0 * W + j] * (1 - fr) + Kc[(i0 + 1) * W + j] * fr;
      ks[j] = Ks[i0 * W + j] * (1 - fr) + Ks[(i0 + 1) * W + j] * fr;
    }
  }
};

// ============================ cepstral smoothing ============================

struct Cepstrum {
  int G = 256, Q = 30;
  std::vector<double> ct;

  void init(int G_, int Q_) {
    G = G_;
    Q = Q_;
    ct.assign(G * Q, 0.0);
    for (int g = 0; g < G; ++g)
      for (int q = 0; q < Q; ++q)
        ct[g * Q + q] = std::cos(PI / G * (g + 0.5) * q);
  }

  void forward(const std::vector<double> &x, std::vector<double> &c) const {
    c.assign(Q, 0.0);
    for (int q = 0; q < Q; ++q) {
      double s = 0;
      for (int g = 0; g < G; ++g)
        s += x[g] * ct[g * Q + q];
      c[q] = s;
    }
  }

  void inverse(const std::vector<double> &c, std::vector<double> &x) const {
    x.assign(G, 0.0);
    for (int g = 0; g < G; ++g) {
      double s = c[0] / G;
      for (int q = 1; q < Q; ++q)
        s += 2.0 / G * c[q] * ct[g * Q + q];
      x[g] = s;
    }
  }

  void smooth(std::vector<double> &x) const {
    std::vector<double> c;
    forward(x, c);
    inverse(c, x);
  }
};

inline double binToGrid(double bin, int M, int G) {
  return bin * (G - 1) / double(M - 1);
}

inline double gridLookup(const std::vector<double> &g, double gi) {
  int G = (int)g.size();
  double x = clampd(gi, 0.0, G - 1.000001);
  int i = (int)x;
  double f = x - i;
  return g[i] * (1 - f) + g[i + 1] * f;
}

// ============================== object model ================================

struct Harmonic {
  int h = 0;
  double bin = 0.0;
  double amp = 0.0;
  double phase = 0.0;
  bool ok = true;
  double bound = 0.0;
};

struct Observation {
  double f0 = 0.0;
  double B = 0.0;
  int q = 1;
  std::vector<double> pattern;
  std::vector<double> cep;
  double maskedFrac = 0.0;
  double f0raw = 0.0;
  std::vector<Harmonic> hs;
  double energy = 0.0;
};

struct Track {
  int id = -1;
  double f0 = 0.0;
  int lastFrame = -1000;
  bool hasCep = false;
  std::vector<double> cep;
  bool hasPhase = false;
  double alpha = 0.0, beta = 0.0;
  double B = 0.0;
  bool hasB = false;
  int q = 1;
  std::vector<double> pattern;
  std::vector<double> extPhase;
  std::vector<double> jitA, jitF;
  int extFrame = -1000;
  double f0Var = 0.0, ampVar = 0.0, prevF0 = 0.0, prevLevel = 0.0;
  bool hasVar = false;
};

// ============================ cut-off detection =============================

static int detectCutoffFromAcc(MCLT &t, const std::vector<double> &acc, int frames, double sr) {
  int M = t.M;
  if (frames <= 0 || acc.empty())
    return M;
  double total = 0;
  for (int k = 0; k < M; ++k)
    total += acc[k];
  if (total <= 0)
    return M;

  std::vector<double> S(M);
  for (int k = 0; k < M; ++k)
    S[k] = 10.0 * std::log10(acc[k] / frames + 1e-30);

  // Smooth spectrum with a 5-bin moving average
  std::vector<double> Ss(M);
  const int SM = 2;
  for (int k = 0; k < M; ++k) {
    double s = 0;
    int c = 0;
    for (int j = std::max(0, k - SM); j <= std::min(M - 1, k + SM); ++j) {
      s += S[j];
      ++c;
    }
    Ss[k] = s / c;
  }

  // 1. Measure top floor in [20000 Hz, 21500 Hz]
  int k_top_lo = (int)clampd(std::lround(t.hzToBin(20000.0, sr)), 0, M - 1);
  int k_top_hi = (int)clampd(std::lround(t.hzToBin(21500.0, sr)), k_top_lo + 1, M);
  double top_floor = 0.0;
  for (int k = k_top_lo; k < k_top_hi; ++k)
    top_floor += Ss[k];
  top_floor /= std::max(1, k_top_hi - k_top_lo);

  // 2. Measure representative passband level in [3000 Hz, 8000 Hz]
  int k_pass_lo = (int)clampd(std::lround(t.hzToBin(3000.0, sr)), 0, M - 1);
  int k_pass_hi = (int)clampd(std::lround(t.hzToBin(8000.0, sr)), k_pass_lo + 1, M);
  std::vector<double> pass_vals;
  pass_vals.reserve(k_pass_hi - k_pass_lo);
  for (int k = k_pass_lo; k < k_pass_hi; ++k)
    pass_vals.push_back(Ss[k]);
  std::sort(pass_vals.begin(), pass_vals.end());
  double pass_level = pass_vals.empty() ? -30.0 : pass_vals[pass_vals.size() * 3 / 4];

  // 3. If passband energy is well above top floor, scan down from 20 kHz to find cutoff
  const double drop = pass_level - top_floor;
  if (drop > 15.0) {
    const double thr = top_floor + std::min(12.0, drop * 0.25);
    int k_scan_hi = (int)clampd(std::lround(t.hzToBin(20000.0, sr)), 0, M - 1);
    int k_scan_lo = (int)clampd(std::lround(t.hzToBin(8000.0, sr)), 0, M - 1);
    for (int k = k_scan_hi; k >= k_scan_lo + 2; --k) {
      if (Ss[k] > thr && Ss[k - 1] > thr && Ss[k - 2] > thr) {
        // Found the transition frequency where audio content drops into the stopband floor
        return std::min(k + 1, M - 8);
      }
    }
  }

  return M;
}

static int detectCutoffBins(MCLT &t, const std::vector<double> &mono, double sr) {
  int M = t.M, N = t.N;
  if (mono.size() < static_cast<size_t>(N))
    return M;
  std::vector<double> pad(mono.size() + 2 * M, 0.0);
  std::copy(mono.begin(), mono.end(), pad.begin() + M);
  std::vector<double> acc(M, 0.0);
  std::vector<cd> X;
  int frames = 0;
  for (size_t p = 0; p + N <= pad.size(); p += M) {
    t.analyze(&pad[p], X);
    for (int k = 0; k < M; ++k)
      acc[k] += std::norm(X[k]);
    ++frames;
  }
  return detectCutoffFromAcc(t, acc, frames, sr);
}

} // namespace

// ============================== Impl ========================================

struct HFExtender::Impl {
  HFExtenderConfig cfg;
  double sr = 44100.0;
  int channels = 2;
  int M = 1024, N = 2048, G = 256;
  int kcut = 1024, kmax = 1024;
  double detected_cutoff = -1.0;
  bool cutoff_initialized = false;
  bool cutoff_locked = false;
  std::vector<double> spec_acc;
  int spec_frames = 0;
  int highest_active_bin = 0;
  int active_gr_count = 0;
  std::array<double, GRANULE_SAMPLES> peak_spec{};
  std::array<double, GRANULE_SAMPLES> mdct_acc{};

  MCLT mclt;
  Kernel ker;
  Cepstrum cep;

  struct ChanState {
    std::vector<Track> tracks;
    int prevDelta = 0;
    double prevTheta = 0;
  };
  std::vector<ChanState> st;

  struct Slot {
    std::vector<cd> X, Xres;
    std::vector<Observation> obs;
    std::vector<double> resCep;
    std::vector<double> resCost;
    double transientness = 0.0;
    bool valid = false, hasRes = false;
  };
  std::vector<std::vector<Slot>> ring;
  int ringN = 7;
  int nextTrackId = 0;

  std::vector<cd> X, Xres, XhiObj, XhiRes, Xhi;
  std::vector<double> mag;
  std::vector<double> rphase;
  std::vector<double> nfloor;
  uint64_t nrng = 0xB5026F5AA96619E9ull;
  double kernelPeak = 1.0;

  // History buffer for streaming PCM
  std::vector<std::vector<double>> pcm_history;
  size_t pcm_history_len = 0;
  int frame_idx = 0;

  double nrand() {
    nrng ^= nrng << 13;
    nrng ^= nrng >> 7;
    nrng ^= nrng << 17;
    return (double)(nrng >> 11) * (1.0 / 9007199254740992.0);
  }

  static double envNoise(int g) {
    uint64_t v = (uint64_t)(g + 1) * 0x9E3779B97F4A7C15ull;
    v ^= v >> 29;
    v *= 0xBF58476D1CE4E5B9ull;
    v ^= v >> 32;
    double u1 = (double)((v >> 11) & 0x1FFFFF) / 2097152.0;
    v *= 0x94D049BB133111EBull;
    v ^= v >> 31;
    double u2 = (double)((v >> 11) & 0x1FFFFF) / 2097152.0;
    double u3 = std::max(u1, 1e-12);
    return std::sqrt(-2.0 * std::log(u3)) * std::cos(2.0 * PI * u2);
  }

  void init(const HFExtenderConfig &c, double sample_rate, int nch) {
    cfg = c;
    sr = sample_rate;
    channels = nch;
    M = 1024;
    N = 2 * M;
    G = cfg.grid;
    mclt.init(M);
    ker.build(mclt);
    cep.init(G, cfg.cep_q);
    st.assign(channels, ChanState());
    ringN = 2 * cfg.look + 1;
    ring.assign(channels, std::vector<Slot>(ringN));

    cd kc[Kernel::W], ks[Kernel::W];
    ker.get(0.5, kc, ks);
    double p = 0;
    for (int j = 0; j < Kernel::W; ++j)
      p = std::max(p, std::abs(kc[j]));
    kernelPeak = std::max(p, 1e-9);

    rphase.resize(M);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (int k = 0; k < M; ++k) {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      rphase[k] = ((s >> 11) * (1.0 / 9007199254740992.0)) * 2.0 * PI - PI;
    }

    pcm_history_len = 2 * N;
    pcm_history.assign(channels, std::vector<double>(pcm_history_len, 0.0));
    cutoff_initialized = false;
    cutoff_locked = false;
    spec_acc.assign(M, 0.0);
    spec_frames = 0;
    highest_active_bin = 0;
    active_gr_count = 0;
    peak_spec.fill(0.0);
    mdct_acc.fill(0.0);
    kcut = M;
    kmax = M;
    frame_idx = 0;
  }

  bool fitAndPeel(std::vector<cd> &R, double bin, double &amp, double &phase) {
    int k0 = (int)std::floor(bin);
    double d = bin - k0;
    if (k0 - Kernel::R < 0 || k0 + Kernel::R >= M)
      return false;
    cd kc[Kernel::W], ks[Kernel::W];
    ker.get(d, kc, ks);
    double pp = 0, pq = 0, qq = 0, pr = 0, qr = 0;
    for (int j = 0; j < Kernel::W; ++j) {
      cd P = kc[j], Q = -ks[j], Rv = R[k0 - Kernel::R + j];
      pp += P.real() * P.real() + P.imag() * P.imag();
      qq += Q.real() * Q.real() + Q.imag() * Q.imag();
      pq += P.real() * Q.real() + P.imag() * Q.imag();
      pr += P.real() * Rv.real() + P.imag() * Rv.imag();
      qr += Q.real() * Rv.real() + Q.imag() * Rv.imag();
    }
    double det = pp * qq - pq * pq;
    if (std::fabs(det) < 1e-18)
      return false;
    double a = (qq * pr - pq * qr) / det;
    double b = (-pq * pr + pp * qr) / det;
    for (int j = 0; j < Kernel::W; ++j)
      R[k0 - Kernel::R + j] -= a * kc[j] - b * ks[j];
    amp = std::sqrt(a * a + b * b);
    phase = std::atan2(b, a);
    return true;
  }

  void addAtom(std::vector<cd> &Y, double bin, double amp, double phase, int kfrom) {
    int k0 = (int)std::floor(bin);
    double d = bin - k0;
    if (k0 - Kernel::R < 0 || k0 + Kernel::R >= M)
      return;
    cd kc[Kernel::W], ks[Kernel::W];
    ker.get(d, kc, ks);
    double a = amp * std::cos(phase), b = amp * std::sin(phase);
    for (int j = 0; j < Kernel::W; ++j) {
      int k = k0 - Kernel::R + j;
      if (k < kfrom || k >= kmax)
        continue;
      Y[k] += a * kc[j] - b * ks[j];
    }
  }

  double findF0(const std::vector<double> &m, double &bestScore) {
    double lo = std::log(55.0), hi = std::log(std::min(1000.0, sr * 0.25));
    int steps = 600;
    double best = 0;
    bestScore = 0;
    for (int i = 0; i <= steps; ++i) {
      double f0 = std::exp(lo + (hi - lo) * i / steps);
      double s = 0;
      int used = 0;
      for (int h = 1; h <= 16; ++h) {
        double bin = mclt.hzToBin(h * f0, sr);
        if (bin >= kcut - 2)
          break;
        int k = (int)std::lround(bin);
        if (k < 1 || k >= M - 1)
          break;
        double v = std::max(m[k], std::max(m[k - 1], m[k + 1]));
        s += v / std::sqrt((double)h);
        ++used;
      }
      if (used < 3)
        continue;
      if (s > bestScore) {
        bestScore = s;
        best = f0;
      }
    }
    return best;
  }

  double refineBin(const std::vector<cd> &R, double pred, double tol = 1.2) {
    int k = (int)std::lround(pred);
    int rad = std::max(1, (int)std::ceil(tol));
    int kb = k;
    double bv = -1;
    for (int t = k - rad; t <= k + rad; ++t) {
      if (t < 1 || t >= M - 1)
        continue;
      double v = std::abs(R[t]);
      if (v > bv) {
        bv = v;
        kb = t;
      }
    }
    if (kb < 1 || kb >= M - 1)
      return pred;
    double a = std::abs(R[kb - 1]), b = std::abs(R[kb]), c = std::abs(R[kb + 1]);
    double den = a - 2 * b + c;
    double dl = (std::fabs(den) < 1e-20) ? 0.0 : 0.5 * (a - c) / den;
    dl = clampd(dl, -0.5, 0.5);
    double out = kb + dl;
    if (std::fabs(out - pred) > tol)
      return pred;
    return out;
  }

  void computeNoiseFloor(const std::vector<cd> &R) {
    const int HW = 16;
    nfloor.assign(M, 0.0);
    std::vector<double> tmpv;
    int step = 4;
    for (int k = 0; k < M; k += step) {
      int a = std::max(0, k - HW), b = std::min(M, k + HW);
      tmpv.clear();
      for (int j = a; j < b; ++j)
        tmpv.push_back(std::abs(R[j]));
      double v = 0;
      if (tmpv.size() >= 4) {
        size_t q = (size_t)(tmpv.size() * 0.5);
        if (q >= tmpv.size())
          q = tmpv.size() - 1;
        std::nth_element(tmpv.begin(), tmpv.begin() + q, tmpv.end());
        v = tmpv[q] * 1.20;
      }
      for (int j = k; j < std::min(M, k + step); ++j)
        nfloor[j] = v;
    }
  }

  double floorAt(double bin) const {
    int k = (int)clampd(std::lround(bin), 0, M - 1);
    return (nfloor.empty() ? 0.0 : nfloor[k]);
  }

  void markReliable(Harmonic &h) const {
    double f = floorAt(h.bin);
    h.bound = f * 2.0 / kernelPeak;
    h.ok = (h.amp > h.bound);
  }

  static inline double partialHz(double f0, double B, int h) {
    return h * f0 * std::sqrt(1.0 + B * (double)h * h);
  }

  void fitInharmonic(const std::vector<Harmonic> &hs, double &f0, double &B) {
    int n = (int)hs.size();
    if (!cfg.inharm || n < 8) {
      B = 0;
      return;
    }
    double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    int hmax = 0, nok = 0;
    for (int i = 0; i < n; ++i)
      if (hs[i].ok)
        ++nok;
    bool useOk = (nok >= 8);
    for (int i = 0; i < n; ++i) {
      if (useOk && !hs[i].ok)
        continue;
      double h = hs[i].h;
      double F = mclt.binToHz(hs[i].bin, sr);
      double w = hs[i].amp * hs[i].amp;
      if (!(w > 0) || F <= 0)
        continue;
      w *= (h * h * h * h) / (F * F);
      double y = (F / h) * (F / h), x = h * h;
      sw += w;
      sx += w * x;
      sy += w * y;
      sxx += w * x * x;
      sxy += w * x * y;
      hmax = std::max(hmax, hs[i].h);
    }
    if (sw <= 0 || hmax < 12) {
      B = 0;
      return;
    }
    double den = sw * sxx - sx * sx;
    if (std::fabs(den) < 1e-12) {
      B = 0;
      return;
    }
    double c1 = (sw * sxy - sx * sy) / den;
    double c0 = (sy - c1 * sx) / sw;
    if (!(c0 > 0)) {
      B = 0;
      return;
    }
    double f0n = std::sqrt(c0), Bn = c1 / c0;
    if (!std::isfinite(f0n) || !std::isfinite(Bn)) {
      B = 0;
      return;
    }
    Bn = clampd(Bn, 0.0, 5e-3);
    if (f0n > 44.0 && f0n < 1400.0) {
      f0 = f0n;
      B = Bn;
    } else
      B = 0;
  }

  void probeHarmonics(const std::vector<cd> &R, double &f0, double &B,
                      std::vector<Harmonic> &out) {
    out.clear();
    std::vector<cd> trial = R;
    double tol = clampd(0.30 * f0 * mclt.N / sr, 1.2, 2.5);
    double f0c = f0, Bc = B;
    for (int h = 1; h <= 200; ++h) {
      double pred = mclt.hzToBin(partialHz(f0c, Bc, h), sr);
      if (pred < Kernel::R + 1 || pred >= kcut - Kernel::R - 1)
        break;
      double bin = refineBin(trial, pred, tol);
      double a, ph;
      if (!fitAndPeel(trial, bin, a, ph))
        continue;
      Harmonic hh{h, bin, a, ph, true, 0.0};
      markReliable(hh);
      out.push_back(hh);
      if ((int)out.size() >= 10 && h % 3 == 0) {
        double f0n = f0c, Bn = Bc;
        fitInharmonic(out, f0n, Bn);
        if (f0n > 0 && std::fabs(f0n / f0c - 1.0) < 0.03) {
          f0c = f0n;
          Bc = Bn;
        }
      }
    }
    f0 = f0c;
    B = Bc;
  }

  void refineF0B(const std::vector<double> &m, double &f0, double &B) {
    B = 0;
    if (!cfg.inharm)
      return;
    static const double Bcand[] = {0.0, 2e-5, 1e-4, 3e-4, 1e-3};
    int nB = (int)(sizeof(Bcand) / sizeof(Bcand[0]));
    double bf = f0, bb = 0, best = -1, best0 = -1, bf0 = f0;
    for (int bi = 0; bi < nB; ++bi) {
      for (int fi = -5; fi <= 5; ++fi) {
        double f = f0 * (1.0 + fi * 0.005);
        double s = 0;
        int used = 0;
        for (int h = 1; h <= 80; ++h) {
          double bin = mclt.hzToBin(partialHz(f, Bcand[bi], h), sr);
          if (bin >= kcut - 2)
            break;
          int k = (int)std::lround(bin);
          if (k < 1 || k >= M - 1)
            break;
          s += std::max(m[k], std::max(m[k - 1], m[k + 1])) / std::sqrt((double)h);
          ++used;
        }
        if (used < 6)
          continue;
        if (s > best) {
          best = s;
          bf = f;
          bb = Bcand[bi];
        }
        if (bi == 0 && s > best0) {
          best0 = s;
          bf0 = f;
        }
      }
    }
    if (best > 0 && best > best0 * 1.02) {
      f0 = bf;
      B = bb;
    } else if (best0 > 0) {
      f0 = bf0;
      B = 0.0;
    }
  }

  void extractObjects(std::vector<cd> &R, std::vector<Observation> &obs) {
    obs.clear();
    double lowEnergy = 0;
    for (int k = 0; k < kcut; ++k)
      lowEnergy += std::norm(R[k]);
    if (lowEnergy <= 1e-20)
      return;

    computeNoiseFloor(R);
    mag.assign(M, 0.0);
    for (int o = 0; o < 3; ++o) {
      for (int k = 0; k < M; ++k)
        mag[k] = std::abs(R[k]);
      double score = 0;
      double f0 = findF0(mag, score);
      if (f0 <= 0)
        break;

      Observation ob;
      ob.f0 = f0;
      ob.B = 0;
      refineF0B(mag, ob.f0, ob.B);
      probeHarmonics(R, ob.f0, ob.B, ob.hs);
      if ((int)ob.hs.size() < 3)
        break;

      ob.hs.clear();
      ob.energy = 0;
      for (int h = 1; h <= 200; ++h) {
        double pred = mclt.hzToBin(partialHz(ob.f0, ob.B, h), sr);
        if (pred < Kernel::R + 1 || pred >= kcut - Kernel::R - 1)
          break;
        double bin = refineBin(R, pred);
        double a, ph;
        if (!fitAndPeel(R, bin, a, ph))
          continue;
        Harmonic hh{h, bin, a, ph, true, 0.0};
        markReliable(hh);
        ob.hs.push_back(hh);
        ob.energy += a * a;
      }
      if ((int)ob.hs.size() >= 3)
        obs.push_back(ob);
      else
        break;
    }
  }

  Track *matchTrack(ChanState &cs, double f0, int frame) {
    Track *best = nullptr;
    double bestd = 1e9;
    for (auto &t : cs.tracks) {
      if (frame - t.lastFrame > 8)
        continue;
      double r = std::fabs(std::log(f0 / std::max(t.f0, 1e-9)));
      if (r < 0.045 && r < bestd) {
        bestd = r;
        best = &t;
      }
    }
    if (best)
      return best;
    Track t;
    t.id = nextTrackId++;
    t.f0 = f0;
    t.lastFrame = frame;
    cs.tracks.push_back(t);
    if (cs.tracks.size() > 32) {
      cs.tracks.erase(std::remove_if(cs.tracks.begin(), cs.tracks.end(),
                                     [&](const Track &x) {
                                       return frame - x.lastFrame > 300;
                                     }),
                      cs.tracks.end());
    }
    return &cs.tracks.back();
  }

  void smoothKnown(std::vector<double> &env, int gc) {
    if (gc < 8)
      return;
    std::vector<double> t(G);
    for (int g = 0; g < G; ++g) {
      double x = (double)g * (gc - 1) / (G - 1);
      int i = (int)x;
      if (i > gc - 2)
        i = gc - 2;
      double fr = x - i;
      t[g] = env[i] * (1 - fr) + env[i + 1] * fr;
    }
    cep.smooth(t);
    for (int g = 0; g < gc; ++g) {
      double x = (double)g * (G - 1) / (gc - 1);
      int i = (int)x;
      if (i > G - 2)
        i = G - 2;
      double fr = x - i;
      env[g] = t[i] * (1 - fr) + t[i + 1] * fr;
    }
  }

  static bool levinson(const std::vector<double> &r, int p, std::vector<double> &a) {
    a.assign(p + 1, 0.0);
    a[0] = 1.0;
    double e = r[0];
    if (!(e > 1e-30))
      return false;
    std::vector<double> t(p + 1, 0.0);
    for (int i = 1; i <= p; ++i) {
      double acc = r[i];
      for (int j = 1; j < i; ++j)
        acc += a[j] * r[i - j];
      double k = -acc / e;
      if (!std::isfinite(k) || std::fabs(k) >= 0.999)
        k = (k > 0 ? 0.999 : -0.999);
      t = a;
      for (int j = 1; j < i; ++j)
        a[j] = t[j] + k * t[i - j];
      a[i] = k;
      e *= (1.0 - k * k);
      if (!(e > 1e-30))
        return i > 2;
    }
    return true;
  }

  void extrapolateEnv(std::vector<double> &env, int gc, double slopeLo = -5.0,
                      double slopeHi = -0.15, double slopeOverride = 1e30,
                      double interOverride = 1e30) {
    gc = (int)clampd(gc, 0, G);
    if (gc < 12) {
      double v = (gc > 0) ? env[gc - 1] : -20.0;
      for (int g = gc; g < G; ++g)
        env[g] = v;
      return;
    }
    int g0 = std::max(4, (int)(gc * 0.35));
    double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int g = g0; g < gc; ++g) {
      double w = 1.0 + 3.0 * (g - g0) / std::max(1, gc - 1 - g0);
      double lx = std::log(g + 1.0);
      sw += w;
      sx += w * lx;
      sy += w * env[g];
      sxx += w * lx * lx;
      sxy += w * lx * env[g];
    }
    double den = sw * sxx - sx * sx;
    double b = (std::fabs(den) > 1e-12) ? (sw * sxy - sx * sy) / den : -1.5;
    if (slopeOverride < 1e29)
      b = slopeOverride;
    b = clampd(b, slopeLo, slopeHi);
    bool haveInter = (interOverride < 1e29 && slopeOverride < 1e29);
    double a0 = haveInter ? interOverride : (sy - b * sx) / sw;
    auto trend = [&](double g) { return a0 + b * std::log(g + 1.0); };

    std::vector<double> d(G, 0.0);
    double mean = 0;
    for (int g = 0; g < gc; ++g) {
      d[g] = env[g] - trend(g);
      mean += d[g];
    }
    mean /= gc;
    for (int g = 0; g < gc; ++g)
      d[g] -= mean;
    double var = 0;
    for (int g = 0; g < gc; ++g)
      var += d[g] * d[g];
    double sd = std::sqrt(var / gc);
    if (!(sd > 1e-9)) {
      for (int g = gc; g < G; ++g)
        env[g] = trend(g) + mean;
      return;
    }

    int p = (cfg.env_order > 0) ? cfg.env_order : std::min(40, std::max(4, gc / 3));
    p = std::min(p, std::max(4, gc / 2));
    std::vector<double> r(p + 1, 0.0);
    for (int k = 0; k <= p; ++k) {
      double s = 0;
      for (int n = k; n < gc; ++n)
        s += d[n] * d[n - k];
      r[k] = s;
    }
    r[0] *= 1.0001;
    r[0] += 1e-12;
    std::vector<double> a;
    bool ok = levinson(r, p, a);

    int Tb = 0;
    double rb = 0;
    for (int T = 4; T <= gc / 2; ++T) {
      double num = 0, e1 = 0, e2 = 0;
      for (int n = T; n < gc; ++n) {
        num += d[n] * d[n - T];
        e1 += d[n] * d[n];
        e2 += d[n - T] * d[n - T];
      }
      double rr = num / std::sqrt(e1 * e2 + 1e-30);
      if (rr > rb) {
        rb = rr;
        Tb = T;
      }
    }
    double wlt = clampd((rb - 0.25) / 0.55, 0.0, 0.85);

    double sigma = 0;
    {
      int start = std::max(p, Tb) + 1;
      double acc = 0;
      int cnt = 0;
      for (int g = start; g < gc; ++g) {
        double vAR = 0;
        if (ok)
          for (int k = 1; k <= p; ++k)
            vAR -= a[k] * d[g - k];
        double vLT = (Tb > 0) ? d[g - Tb] : 0.0;
        double pr = wlt * vLT + (1.0 - wlt) * vAR;
        if (!std::isfinite(pr))
          pr = 0;
        double e = d[g] - pr;
        acc += e * e;
        ++cnt;
      }
      if (cnt > 4)
        sigma = std::sqrt(acc / cnt);
      sigma = clampd(sigma, 0.0, 1.5 * sd);
    }

    const double rho = cfg.env_rho;
    if (ok) {
      double s = 1.0;
      for (int k = 1; k <= p; ++k) {
        s *= rho;
        a[k] *= s;
      }
    }
    double lim = 2.2 * sd;

    for (int g = gc; g < G; ++g) {
      double vAR = 0;
      if (ok) {
        for (int k = 1; k <= p; ++k)
          vAR -= a[k] * d[g - k];
      }
      if (!std::isfinite(vAR))
        vAR = 0;
      double vLT = (Tb > 0 && g - Tb >= 0) ? d[g - Tb] : 0.0;
      double v = wlt * vLT + (1.0 - wlt) * vAR;
      if (!std::isfinite(v))
        v = 0;
      v = clampd(v, -lim, lim);
      double e = clampd(cfg.env_innov * sigma * envNoise(g), -lim, lim);
      double fade = 0.80 + 0.20 * std::exp(-(double)(g - gc) / (4.0 * std::max(8, gc)));
      d[g] = v;
      env[g] = trend(g) + mean + (v * fade + e) * cfg.struct_amt;
    }
  }

  void buildObjectEnvelope(Observation &ob, std::vector<double> &env) {
    env.assign(G, -20.0);
    int n = (int)ob.hs.size();
    std::vector<double> gx, gy;
    for (int i = 0; i < n; ++i) {
      if (ob.hs[i].amp <= 0)
        continue;
      gx.push_back(binToGrid(ob.hs[i].bin, M, G));
      gy.push_back(std::log(std::max(ob.hs[i].amp, 1e-9)));
    }
    if (gx.empty())
      return;

    double trendA = 1e30, trendB = 1e30;
    {
      int nok = 0;
      for (int i = 0; i < n; ++i)
        if (ob.hs[i].ok && ob.hs[i].amp > 0)
          ++nok;
      if (nok >= 5) {
        double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
        int seen = 0;
        for (int i = 0; i < n; ++i) {
          if (!ob.hs[i].ok || ob.hs[i].amp <= 0)
            continue;
          ++seen;
          double w = 1.0 + 3.0 * (double)seen / nok;
          double g2 = binToGrid(ob.hs[i].bin, M, G);
          double lx = std::log(g2 + 1.0);
          double ly = std::log(std::max(ob.hs[i].amp, 1e-9));
          sw += w;
          sx += w * lx;
          sy += w * ly;
          sxx += w * lx * lx;
          sxy += w * lx * ly;
        }
        double den = sw * sxx - sx * sx;
        if (std::fabs(den) > 1e-12) {
          trendB = (sw * sxy - sx * sy) / den;
          double aa = 0, an = 0;
          int seen2 = 0;
          for (int i = 0; i < n; ++i) {
            if (!ob.hs[i].ok || ob.hs[i].amp <= 0)
              continue;
            ++seen2;
            if (seen2 < (int)(nok * 0.6))
              continue;
            double g2 = binToGrid(ob.hs[i].bin, M, G);
            aa += std::log(std::max(ob.hs[i].amp, 1e-9)) - trendB * std::log(g2 + 1.0);
            an += 1.0;
          }
          trendA = (an > 0) ? aa / an : (sy - trendB * sx) / sw;
        }
      }
    }

    int gc = (int)std::floor(gx.back()) + 1;
    gc = (int)clampd(gc, 1, G);
    for (int g = 0; g < gc; ++g) {
      double x = g;
      if (x <= gx.front()) {
        env[g] = gy.front();
        continue;
      }
      if (x >= gx.back()) {
        env[g] = gy.back();
        continue;
      }
      size_t i = 1;
      while (i < gx.size() && gx[i] < x)
        ++i;
      double t = (x - gx[i - 1]) / std::max(gx[i] - gx[i - 1], 1e-9);
      env[g] = gy[i - 1] * (1 - t) + gy[i] * t;
    }
    smoothKnown(env, gc);
    extrapolateEnv(env, gc, -5.0, -0.15, trendB, trendA);
    ob.q = 1;
    ob.pattern.assign(1, 0.0);
  }

  void fitPhaseModel(const Observation &ob, Track &tr) {
    int n = (int)ob.hs.size();
    if (n < 2)
      return;
    double pa = tr.alpha, pb = tr.beta;
    bool havePrior = tr.hasPhase;
    if (havePrior)
      pa += PI * ob.f0 * N / sr;

    std::vector<double> up(n);
    if (havePrior) {
      for (int i = 0; i < n; ++i) {
        double pred = pa * ob.hs[i].h + pb;
        up[i] = pred + wrapPi(ob.hs[i].phase - pred);
      }
    } else {
      up[0] = ob.hs[0].phase;
      for (int i = 1; i < n; ++i) {
        double step = (double)(ob.hs[i].h - ob.hs[i - 1].h);
        double guess = up[i - 1] + (i >= 2 ? (up[i - 1] - up[i - 2]) : 0.0) * step;
        up[i] = guess + wrapPi(ob.hs[i].phase - guess);
      }
    }
    double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; ++i) {
      double w = ob.hs[i].amp * ob.hs[i].amp + 1e-12;
      double x = ob.hs[i].h;
      sw += w;
      sx += w * x;
      sy += w * up[i];
      sxx += w * x * x;
      sxy += w * x * up[i];
    }
    double den = sw * sxx - sx * sx;
    if (std::fabs(den) < 1e-12)
      return;
    tr.alpha = (sw * sxy - sx * sy) / den;
    tr.beta = (sy - tr.alpha * sx) / sw;
    tr.hasPhase = true;
  }

  void extendObject(Observation &ob, Track &tr, std::vector<double> &env, int frameNow,
                    std::vector<cd> &Y) {
    int hStart = ob.hs.back().h + 1;
    double f0 = ob.f0;
    double gain = dbToLin(cfg.obj_gain_db);
    double fcut = mclt.binToHz(kcut, sr);
    for (int h = hStart;; ++h) {
      double hz = partialHz(f0, tr.B, h);
      double bin = mclt.hzToBin(hz, sr);
      if (bin >= kmax - Kernel::R - 1)
        break;
      if (h > 1500)
        break;
      double a = std::exp(gridLookup(env, binToGrid(bin, M, G)));
      if (cfg.tilt_db_oct != 0.0 && hz > fcut)
        a *= dbToLin(cfg.tilt_db_oct * std::log2(hz / fcut));
      a *= gain;
      if (!(a > 1e-9))
        continue;
      if ((int)tr.extPhase.size() <= h)
        tr.extPhase.resize(h + 1, 1e30);
      double ph;
      if (tr.extFrame == frameNow - 1 && tr.extPhase[h] < 1e29)
        ph = wrapPi(tr.extPhase[h] + PI * hz * N / sr);
      else
        ph = wrapPi(tr.alpha * h + tr.beta);
      tr.extPhase[h] = ph;
      addAtom(Y, bin, a, ph, kcut);
    }
    tr.extFrame = frameNow;
  }

  static constexpr int NTHETA = 8;
  void residualDeltas(std::vector<int> &deltas) const {
    deltas.clear();
    for (double r : {0.5, 0.4, 0.3, 0.25, 0.2, 0.15}) {
      int d = (int)std::lround(kcut * r);
      if (d >= 16 && kcut - 2 * d >= 8)
        deltas.push_back(d);
    }
    if (deltas.empty())
      deltas.push_back(std::max(16, kcut / 4));
  }

  void residualSearch(const std::vector<cd> &R, const std::vector<int> &deltas,
                      std::vector<double> &cost) {
    int nD = (int)deltas.size();
    cost.assign((size_t)nD * NTHETA, 1e3);
    for (int di = 0; di < nD; ++di) {
      int D = deltas[di];
      int vt = kcut - D, vs = kcut - 2 * D;
      if (vs < 0)
        continue;
      double ee = 0, tt = 0;
      for (int j = 0; j < D; ++j) {
        ee += std::norm(R[vs + j]);
        tt += std::norm(R[vt + j]);
      }
      if (ee < 1e-24 || tt < 1e-24)
        continue;
      double g = std::sqrt(tt / ee);
      for (int ti = 0; ti < NTHETA; ++ti) {
        cd rot = std::polar(g, 2 * PI * ti / NTHETA);
        double c = 0;
        for (int j = 0; j < D; ++j)
          c += std::norm(R[vt + j] - rot * R[vs + j]);
        cost[(size_t)di * NTHETA + ti] = c / tt;
      }
    }
  }

  bool residualEnvCep(const std::vector<cd> &R, std::vector<double> &out) {
    std::vector<double> env;
    if (!residualEnv(R, env))
      return false;
    cep.forward(env, out);
    return true;
  }

  bool residualEnv(const std::vector<cd> &R, std::vector<double> &envOut) {
    if (kcut < 32)
      return false;
    const int BW = 8, HW = 12;
    std::vector<double> gx, gy, tmpv;
    for (int k = 0; k + BW <= kcut; k += BW) {
      int a = std::max(0, k + BW / 2 - HW), b = std::min(kcut, k + BW / 2 + HW);
      tmpv.clear();
      for (int j = a; j < b; ++j)
        tmpv.push_back(std::abs(R[j]));
      if (tmpv.size() < 4)
        continue;
      size_t q = (size_t)(tmpv.size() * 0.5);
      if (q >= tmpv.size())
        q = tmpv.size() - 1;
      std::nth_element(tmpv.begin(), tmpv.begin() + q, tmpv.end());
      double e = tmpv[q] * 1.20;
      gx.push_back(binToGrid(k + BW * 0.5, M, G));
      gy.push_back(std::log(std::max(e, 1e-10)));
    }
    if (gx.size() < 4)
      return false;
    std::vector<double> env(G, 0.0);
    int gc = (int)std::floor(gx.back()) + 1;
    gc = (int)clampd(gc, 1, G);
    for (int g = 0; g < gc; ++g) {
      double x = g;
      if (x <= gx.front()) {
        env[g] = gy.front();
        continue;
      }
      if (x >= gx.back()) {
        env[g] = gy.back();
        continue;
      }
      size_t i = 1;
      while (i < gx.size() && gx[i] < x)
        ++i;
      double t = (x - gx[i - 1]) / std::max(gx[i] - gx[i - 1], 1e-9);
      env[g] = gy[i - 1] * (1 - t) + gy[i] * t;
    }
    smoothKnown(env, gc);
    extrapolateEnv(env, gc, -5.0, 0.10);
    envOut.swap(env);
    return true;
  }

  void extendResidual(const std::vector<cd> &Src, const std::vector<double> &env,
                      int resDelta, double resTheta, int frame, std::vector<cd> &Y) {
    if (kcut < 32 || (int)env.size() != G)
      return;
    int D = resDelta;
    std::vector<cd> tmp(kmax, cd(0, 0));
    for (int k = kcut; k < kmax; ++k) {
      int off = k - kcut;
      int r = off / D + 1;
      int src = kcut - D + (off % D);
      if (src < 0 || src >= kcut)
        continue;
      cd rot = std::polar(1.0, resTheta * r + PI * (double)(r * D) * frame + rphase[k]);
      cd v = Src[src] * rot;
      tmp[k] = v;
    }
    double gain = dbToLin(cfg.res_gain_db);
    double fcut = mclt.binToHz(kcut, sr);
    const int BW = 8;
    for (int k0 = kcut; k0 < kmax; k0 += BW) {
      int k1 = std::min(k0 + BW, kmax);
      double e = 0, tgt = 0;
      for (int k = k0; k < k1; ++k) {
        e += std::norm(tmp[k]);
        tgt += std::exp(2.0 * gridLookup(env, binToGrid(k, M, G)));
      }
      if (e < 1e-30)
        continue;
      double g = std::sqrt(tgt / e) * gain;
      double hz = mclt.binToHz(0.5 * (k0 + k1), sr);
      if (cfg.tilt_db_oct != 0.0 && hz > fcut)
        g *= dbToLin(cfg.tilt_db_oct * std::log2(hz / fcut));
      for (int k = k0; k < k1; ++k)
        Y[k] += tmp[k] * g;
    }
  }

  void analyzeFrame(const double *x, Slot &s) {
    s.obs.clear();
    s.hasRes = false;
    mclt.analyze(x, s.X);
    if (kcut >= kmax || kcut >= M) {
      s.valid = true;
      s.Xres = s.X;
      return;
    }
    s.Xres = s.X;
    extractObjects(s.Xres, s.obs);
    for (auto &ob : s.obs) {
      std::vector<double> env;
      buildObjectEnvelope(ob, env);
      cep.forward(env, ob.cep);
    }
    s.hasRes = residualEnvCep(s.Xres, s.resCep);
    if (s.hasRes) {
      std::vector<int> dl;
      residualDeltas(dl);
      residualSearch(s.Xres, dl, s.resCost);
    } else {
      s.resCost.clear();
    }
    s.valid = true;
  }

  void synthesizeHF(int chan, Slot &c0, int frame, std::vector<double> &hfReal) {
    hfReal.assign(M, 0.0);
    if (!c0.valid || kcut >= kmax || kcut >= M)
      return;
    XhiObj.assign(M, cd(0, 0));
    XhiRes.assign(M, cd(0, 0));
    Xhi.assign(M, cd(0, 0));
    ChanState &cs = st[chan];
    std::vector<double> env;
    for (auto &ob : c0.obs) {
      Track *tr = matchTrack(cs, ob.f0, frame);
      tr->f0 = ob.f0;
      tr->lastFrame = frame;
      fitPhaseModel(ob, *tr);
      cep.inverse(ob.cep, env);
      extendObject(ob, *tr, env, frame, XhiObj);
      cep.forward(env, tr->cep);
      tr->hasCep = true;
    }
    if (c0.hasRes) {
      cep.inverse(c0.resCep, env);
      std::vector<int> dl;
      residualDeltas(dl);
      int nS = (int)dl.size() * NTHETA, stt = 0;
      (void)nS;
      if (!c0.resCost.empty())
        stt = (int)(std::min_element(c0.resCost.begin(), c0.resCost.end()) -
                    c0.resCost.begin());
      int D = dl[stt / NTHETA];
      double th = 2 * PI * (stt % NTHETA) / NTHETA;
      extendResidual(c0.Xres, env, D, th, frame, XhiRes);
    }
    for (int k = kcut; k < kmax; ++k) {
      Xhi[k] = XhiObj[k] + XhiRes[k];
      hfReal[k] = Xhi[k].real();
    }
  }
};

HFExtender::HFExtender() : impl_(std::make_unique<Impl>()) {}
HFExtender::~HFExtender() = default;

void HFExtender::init(const HFExtenderConfig &cfg, double sample_rate, int channels) {
  impl_->init(cfg, sample_rate, channels);
}

double HFExtender::detected_cutoff_hz() const {
  return impl_->detected_cutoff;
}

int HFExtender::cutoff_sfb(int samplerate_index) const {
  if (impl_->detected_cutoff >= 20000.0)
    return NUM_SFB_LONG;
  double fcut = (impl_->detected_cutoff > 0.0)
                    ? impl_->detected_cutoff
                    : impl_->mclt.binToHz(impl_->kcut, impl_->sr);
  const auto &band_table = tables::scale_fact_band_index[samplerate_index];
  for (int sfb = 0; sfb < NUM_SFB_LONG; ++sfb) {
    double f_mid = 0.5 * (band_table[sfb] + band_table[sfb + 1]) * impl_->sr /
                   (2.0 * GRANULE_SAMPLES);
    if (f_mid >= fcut)
      return sfb;
  }
  return NUM_SFB_LONG - 1;
}

void HFExtender::process_frame(
    const std::int16_t *pcm_interleaved, size_t num_samples, int channels,
    int granules_per_frame,
    std::array<std::array<std::array<std::int32_t, GRANULE_SAMPLES>,
                          MAX_GRANULES>,
               MAX_CHANNELS> &out_mdct_ext,
    const std::array<std::array<std::array<std::int32_t, GRANULE_SAMPLES>,
                                MAX_GRANULES>,
                     MAX_CHANNELS> &ref_mdct) {
  if (!pcm_interleaved || num_samples == 0)
    return;
  const size_t total = num_samples * static_cast<size_t>(channels);
  std::vector<double> buf(total);
  for (size_t i = 0; i < total; ++i) {
    buf[i] = static_cast<double>(pcm_interleaved[i]);
  }
  process_frame(buf.data(), num_samples, channels, granules_per_frame,
                out_mdct_ext, ref_mdct);
}

void HFExtender::process_frame(
    const double *pcm_interleaved, size_t num_samples, int channels,
    int granules_per_frame,
    std::array<std::array<std::array<std::int32_t, GRANULE_SAMPLES>,
                          MAX_GRANULES>,
               MAX_CHANNELS> &out_mdct_ext,
    const std::array<std::array<std::array<std::int32_t, GRANULE_SAMPLES>,
                                MAX_GRANULES>,
                     MAX_CHANNELS> &ref_mdct) {
  if (!pcm_interleaved || num_samples == 0)
    return;

  // 1. Demux PCM into history buffer
  const int nch = std::min(channels, impl_->channels);
  for (int c = 0; c < nch; ++c) {
    auto &hist = impl_->pcm_history[c];
    if (num_samples < hist.size()) {
      std::copy(hist.begin() + num_samples, hist.end(), hist.begin());
      for (size_t i = 0; i < num_samples; ++i) {
        hist[hist.size() - num_samples + i] = pcm_interleaved[i * channels + c];
      }
    } else {
      for (size_t i = 0; i < hist.size(); ++i) {
        hist[i] = pcm_interleaved[(num_samples - hist.size() + i) * channels + c];
      }
    }
  }

  // 2. Online cutoff detection using MDCT energy gradient across frequency
  if (impl_->cfg.cutoff_hz <= 0) {
    for (int gr = 0; gr < granules_per_frame; ++gr) {
      for (int c = 0; c < nch; ++c) {
        const auto &freq = ref_mdct[c][gr];
        double gr_e = 0.0;
        for (int i = 0; i < GRANULE_SAMPLES; ++i) {
          const double v = static_cast<double>(freq[i]);
          gr_e += v * v;
        }
        // Only accumulate active (non-silent) granules
        if (gr_e > 1e6) {
          for (int i = 0; i < GRANULE_SAMPLES; ++i) {
            const double v = static_cast<double>(freq[i]);
            impl_->mdct_acc[i] += v * v;
          }
          ++impl_->active_gr_count;
        }
      }
    }

    if (impl_->active_gr_count >= 2) {
      // 1. Convert accumulated energy to dB
      std::array<double, GRANULE_SAMPLES> db{};
      for (int i = 0; i < GRANULE_SAMPLES; ++i) {
        db[i] = 10.0 * std::log10(impl_->mdct_acc[i] / impl_->active_gr_count + 1e-15);
      }

      // 2. Smooth with 5-bin moving average
      std::array<double, GRANULE_SAMPLES> db_s{};
      for (int i = 0; i < GRANULE_SAMPLES; ++i) {
        double s = 0.0;
        int count = 0;
        for (int d = -2; d <= 2; ++d) {
          if (i + d >= 0 && i + d < GRANULE_SAMPLES) {
            s += db[i + d];
            ++count;
          }
        }
        db_s[i] = s / count;
      }

      // 3. Scan for steep drop (brickwall cutoff) across window W=8 (~300 Hz)
      constexpr int W = 8;
      const int k_min = static_cast<int>(8000.0 * 2.0 * GRANULE_SAMPLES / impl_->sr);
      const int k_max = static_cast<int>(21000.0 * 2.0 * GRANULE_SAMPLES / impl_->sr);
      double best_drop = 0.0;
      int best_k = -1;

      for (int k = k_min + W; k < k_max - W; ++k) {
        double lo = 0.0, hi = 0.0;
        for (int j = k - W; j < k; ++j) lo += db_s[j];
        for (int j = k; j < k + W; ++j) hi += db_s[j];
        lo /= W;
        hi /= W;
        const double drop = lo - hi;
        if (drop > best_drop) {
          best_drop = drop;
          best_k = k;
        }
      }

      // If a steep cliff (drop >= 14 dB) is detected, that is the cutoff!
      if (best_drop >= 14.0 && best_k > 0) {
        // Fine-tune to the knee point where the drop starts
        int k_knee = best_k;
        const double pass_ref = db_s[best_k - W];
        while (k_knee > k_min && db_s[k_knee] < pass_ref - 4.0) {
          --k_knee;
        }
        double f_cut = (k_knee + 0.5) * impl_->sr / (2.0 * GRANULE_SAMPLES);
        impl_->detected_cutoff = f_cut;
        impl_->kcut = (int)clampd(
            std::lround(impl_->mclt.hzToBin(f_cut, impl_->sr)), 64, impl_->M - 8);
      } else {
        // No steep cliff detected: source has natural full bandwidth
        impl_->detected_cutoff = 20000.0;
        impl_->kcut = impl_->M;
      }
      impl_->cutoff_initialized = true;

      static const bool hf_dbg = (std::getenv("RMPS_HF_DEBUG") != nullptr);
      if (hf_dbg && (impl_->active_gr_count == 2 || impl_->active_gr_count % 50 == 0)) {
        std::fprintf(stderr,
                     "[hf_ext] detected cutoff: %.1f Hz (best_drop=%.1f dB at k=%d, kcut=%d, kmax=%d, active_gr=%d)\n",
                     impl_->detected_cutoff, best_drop, best_k, impl_->kcut, impl_->kmax,
                     impl_->active_gr_count);
      }
    }
    double max_hz = (impl_->cfg.max_out_hz > 0) ? impl_->cfg.max_out_hz : 20000.0;
    max_hz = std::min(max_hz, 20000.0);
    impl_->kmax = (int)clampd(
        std::lround(impl_->mclt.hzToBin(max_hz, impl_->sr)) + 1,
        impl_->kcut, impl_->M);
  } else {
    impl_->kcut = (int)clampd(
        std::lround(impl_->mclt.hzToBin(impl_->cfg.cutoff_hz, impl_->sr)), 64,
        impl_->M - 8);
    impl_->detected_cutoff = impl_->cfg.cutoff_hz;
    impl_->cutoff_initialized = true;
    impl_->cutoff_locked = true;
    double max_hz = (impl_->cfg.max_out_hz > 0) ? impl_->cfg.max_out_hz : 20000.0;
    max_hz = std::min(max_hz, 20000.0);
    impl_->kmax = (int)clampd(
        std::lround(impl_->mclt.hzToBin(max_hz, impl_->sr)) + 1,
        impl_->kcut, impl_->M);
  }

  if (impl_->kcut >= impl_->kmax) {
    return;
  }

  // 3. For each granule and channel, analyze and synthesize HF
  const size_t N = impl_->N;
  std::vector<double> mclt_in(N, 0.0);
  std::vector<double> hf_mclt(impl_->M, 0.0);

  for (int gr = 0; gr < granules_per_frame; ++gr) {
    // The granule center in history buffer
    // num_samples is typically 1152 for 2 granules -> 576 samples per granule
    size_t gr_samples = num_samples / granules_per_frame;
    size_t gr_center = impl_->pcm_history_len - num_samples + gr * gr_samples + gr_samples / 2;
    size_t win_start = (gr_center >= N / 2) ? (gr_center - N / 2) : 0;

    for (int c = 0; c < nch; ++c) {
      const auto &hist = impl_->pcm_history[c];
      for (size_t i = 0; i < N; ++i) {
        size_t idx = win_start + i;
        mclt_in[i] = (idx < hist.size()) ? hist[idx] : 0.0;
      }

      Impl::Slot slot;
      impl_->analyzeFrame(mclt_in.data(), slot);
      impl_->synthesizeHF(c, slot, impl_->frame_idx, hf_mclt);

      // Estimate scale factor between MCLT and MP3 MDCT in the reference band [0.5*kcut, 0.9*kcut]
      double energy_ref = 0.0, energy_mclt = 0.0;
      int k_ref_lo = std::max(16, (int)(impl_->kcut * 0.4));
      int k_ref_hi = std::max(k_ref_lo + 8, (int)(impl_->kcut * 0.9));
      for (int k = k_ref_lo; k < k_ref_hi; ++k) {
        double f = impl_->mclt.binToHz(k, impl_->sr);
        int mp3_bin = (int)std::lround(f * GRANULE_SAMPLES * 2.0 / impl_->sr);
        if (mp3_bin >= 0 && mp3_bin < GRANULE_SAMPLES) {
          double v_ref = ref_mdct[c][gr][mp3_bin];
          energy_ref += v_ref * v_ref;
          double v_mclt = slot.X[k].real();
          energy_mclt += v_mclt * v_mclt;
        }
      }
      double scale = 1.0;
      if (energy_mclt > 1e-12 && energy_ref > 1e-12) {
        scale = std::sqrt(energy_ref / energy_mclt);
      }

      // Map MCLT HF lines to MP3 576-line MDCT grid
      int mp3_cut_bin = (int)std::lround(
          impl_->mclt.binToHz(impl_->kcut, impl_->sr) * GRANULE_SAMPLES * 2.0 / impl_->sr);
      mp3_cut_bin = std::clamp(mp3_cut_bin, 0, GRANULE_SAMPLES);

      for (int i = mp3_cut_bin; i < GRANULE_SAMPLES; ++i) {
        double f = (i + 0.5) * impl_->sr / (2.0 * GRANULE_SAMPLES);
        if (f > 20000.0) {
          out_mdct_ext[c][gr][i] = 0;
          continue;
        }
        double mclt_bin = impl_->mclt.hzToBin(f, impl_->sr);
        if (mclt_bin >= impl_->kcut && mclt_bin < impl_->kmax - 1) {
          int k0 = (int)std::floor(mclt_bin);
          double fr = mclt_bin - k0;
          double val = (1.0 - fr) * hf_mclt[k0] + fr * hf_mclt[k0 + 1];
          double scaled_val = val * scale;
          out_mdct_ext[c][gr][i] = fx::sat32(scaled_val);
        } else {
          out_mdct_ext[c][gr][i] = 0;
        }
      }
    }
    ++impl_->frame_idx;
  }
}

} // namespace rmps::post
