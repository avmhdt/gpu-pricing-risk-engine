// GBM path kernels, compiled at run time by CuPy RawModule (NVRTC) and checked by nvcc in CI.
//
// RNG layout (docs/rng.md): key = seed, counter = (draw block, path, stream, replication); one
// Philox call gives 4 float32 (or 2 float64) normals by inverse CDF. Antithetic path p uses
// Philox path p >> 1 and flips the sign for odd p. Every output depends only on
// (seed, stream, replication, path, step), never on the launch configuration.
//
// Reductions (NFR-02): the fused kernels write one float64 (sum, sum of squares) pair per tile
// of kTile = 64 consecutive paths, reduced in a fixed tree order inside the tile. The tile
// results therefore do not depend on block or grid size, and the host adds tiles in index
// order in float64. No floating-point atomics anywhere.
//
// bandwidth_probe and fma_probe (FR-27) arrive in M6.
#include "philox4x32.cuh"

namespace {

constexpr int kTile = 64;

template <typename Real>
struct Rng;

template <>
struct Rng<float> {
  static constexpr int kPerBlock = 4;
  __device__ static void normals(const gpre::Philox4x32Counter& w, float z[4]) {
    for (int i = 0; i < 4; ++i) {
      z[i] = normcdfinvf(gpre::philox_uniform_f32(w.v[i]));
    }
  }
  __device__ static float fma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
  __device__ static float exp(float x) { return expf(x); }
};

template <>
struct Rng<double> {
  static constexpr int kPerBlock = 2;
  __device__ static void normals(const gpre::Philox4x32Counter& w, double z[4]) {
    z[0] = normcdfinv(gpre::philox_uniform_f64(w.v[0], w.v[1]));
    z[1] = normcdfinv(gpre::philox_uniform_f64(w.v[2], w.v[3]));
  }
  __device__ static double fma(double a, double b, double c) { return __fma_rn(a, b, c); }
  __device__ static double exp(double x) { return ::exp(x); }
};

// Simulates one path and calls obs(step, spot) for steps 1..n_steps.
template <typename Real, typename Observer>
__device__ void simulate_path(unsigned long long seed, unsigned int stream,
                              unsigned int replication, long long path, int n_steps,
                              const Real* __restrict__ drift_dt,
                              const Real* __restrict__ vol_sqrt_dt, Real s0, int antithetic,
                              Observer& obs) {
  const gpre::Philox4x32Key key = gpre::philox_key_from_seed(seed);
  const unsigned int rng_path = static_cast<unsigned int>(antithetic ? (path >> 1) : path);
  const Real sign = (antithetic && (path & 1)) ? Real(-1) : Real(1);
  Real x = Real(0);
  Real z[4];
  for (int k = 0; k < n_steps; ++k) {
    const int slot = k % Rng<Real>::kPerBlock;
    if (slot == 0) {
      gpre::Philox4x32Counter ctr;
      ctr.v[0] = static_cast<unsigned int>(k / Rng<Real>::kPerBlock);
      ctr.v[1] = rng_path;
      ctr.v[2] = stream;
      ctr.v[3] = replication;
      Rng<Real>::normals(gpre::philox4x32(ctr, key), z);
    }
    x += Rng<Real>::fma(vol_sqrt_dt[k], sign * z[slot], drift_dt[k]);
    obs(k, s0 * Rng<Real>::exp(x));
  }
}

template <typename Real>
struct StoreObserver {
  Real* out;
  long long path;
  long long n_paths;
  __device__ void operator()(int k, Real s) { out[(k + 1) * n_paths + path] = s; }
};

template <typename Real>
struct TerminalObserver {
  Real last = Real(0);
  __device__ void operator()(int /*k*/, Real s) { last = s; }
};

template <typename Real>
struct AsianObserver {
  const int* mask;
  double sum = 0.0;
  int count = 0;
  __device__ void operator()(int k, Real s) {
    if (mask[k]) {
      sum += static_cast<double>(s);
      ++count;
    }
  }
};

// Fixed-order reduction of one value per thread into per-tile sums. Every thread of the block
// must call it the same number of times (it synchronizes).
__device__ void reduce_tiles(double value, long long tile, long long n_tiles,
                             double* __restrict__ tile_sum, double* __restrict__ tile_sumsq) {
  extern __shared__ double shared[];
  double* sum = shared;
  double* sq = shared + blockDim.x;
  const int t = static_cast<int>(threadIdx.x);
  const int lane = t % kTile;
  sum[t] = value;
  sq[t] = value * value;
  __syncthreads();
  for (int offset = kTile / 2; offset > 0; offset /= 2) {
    if (lane < offset) {
      sum[t] += sum[t + offset];
      sq[t] += sq[t + offset];
    }
    __syncthreads();
  }
  if (lane == 0 && tile < n_tiles) {
    tile_sum[tile] = sum[t];
    tile_sumsq[tile] = sq[t];
  }
  __syncthreads();
}

template <typename Real>
__device__ void european_fused(unsigned long long seed, unsigned int stream,
                               unsigned int replication, int n_paths, int n_steps,
                               const Real* drift_dt, const Real* vol_sqrt_dt, Real s0, Real strike,
                               int is_call, double discount, int antithetic, double* tile_sum,
                               double* tile_sumsq) {
  const long long n_tiles = (static_cast<long long>(n_paths) + kTile - 1) / kTile;
  const long long tiles_per_block = blockDim.x / kTile;
  const long long stride = static_cast<long long>(gridDim.x) * tiles_per_block;
  for (long long base = static_cast<long long>(blockIdx.x) * tiles_per_block; base < n_tiles;
       base += stride) {
    const long long tile = base + threadIdx.x / kTile;
    const long long path = tile * kTile + threadIdx.x % kTile;
    double payoff = 0.0;
    if (path < n_paths) {
      TerminalObserver<Real> obs;
      simulate_path<Real>(seed, stream, replication, path, n_steps, drift_dt, vol_sqrt_dt, s0,
                          antithetic, obs);
      const double st = static_cast<double>(obs.last);
      const double k = static_cast<double>(strike);
      payoff = discount * (is_call ? fmax(st - k, 0.0) : fmax(k - st, 0.0));
    }
    reduce_tiles(payoff, tile, n_tiles, tile_sum, tile_sumsq);
  }
}

template <typename Real>
__device__ void asian_fused(unsigned long long seed, unsigned int stream, unsigned int replication,
                            int n_paths, int n_steps, const Real* drift_dt, const Real* vol_sqrt_dt,
                            Real s0, Real strike, int is_call, const int* fixing_mask,
                            double discount, int antithetic, double* tile_sum, double* tile_sumsq) {
  const long long n_tiles = (static_cast<long long>(n_paths) + kTile - 1) / kTile;
  const long long tiles_per_block = blockDim.x / kTile;
  const long long stride = static_cast<long long>(gridDim.x) * tiles_per_block;
  for (long long base = static_cast<long long>(blockIdx.x) * tiles_per_block; base < n_tiles;
       base += stride) {
    const long long tile = base + threadIdx.x / kTile;
    const long long path = tile * kTile + threadIdx.x % kTile;
    double payoff = 0.0;
    if (path < n_paths) {
      AsianObserver<Real> obs{fixing_mask};
      simulate_path<Real>(seed, stream, replication, path, n_steps, drift_dt, vol_sqrt_dt, s0,
                          antithetic, obs);
      const double avg = obs.sum / static_cast<double>(obs.count > 0 ? obs.count : 1);
      const double k = static_cast<double>(strike);
      payoff = discount * (is_call ? fmax(avg - k, 0.0) : fmax(k - avg, 0.0));
    }
    reduce_tiles(payoff, tile, n_tiles, tile_sum, tile_sumsq);
  }
}

template <typename Real>
__device__ void paths_store(unsigned long long seed, unsigned int stream, unsigned int replication,
                            int n_paths, int n_steps, const Real* drift_dt, const Real* vol_sqrt_dt,
                            Real s0, int antithetic, Real* out) {
  for (long long path = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       path < n_paths; path += static_cast<long long>(gridDim.x) * blockDim.x) {
    out[path] = s0;
    StoreObserver<Real> obs{out, path, n_paths};
    simulate_path<Real>(seed, stream, replication, path, n_steps, drift_dt, vol_sqrt_dt, s0,
                        antithetic, obs);
  }
}

}  // namespace

// Raw Philox output for the stream layout: out[((p * n_blocks_per_path) + j) * 4 + i] = word i
// of Philox((j, p, stream, replication), seed). Used by the bit-exactness tests (AC-W-06).
extern "C" __global__ void philox_uniforms(unsigned long long seed, unsigned int stream,
                                           unsigned int replication, int n_paths,
                                           int n_blocks_per_path, unsigned int* __restrict__ out) {
  const long long total = static_cast<long long>(n_paths) * n_blocks_per_path;
  const gpre::Philox4x32Key key = gpre::philox_key_from_seed(seed);
  for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<long long>(gridDim.x) * blockDim.x) {
    gpre::Philox4x32Counter ctr;
    ctr.v[0] = static_cast<unsigned int>(idx % n_blocks_per_path);
    ctr.v[1] = static_cast<unsigned int>(idx / n_blocks_per_path);
    ctr.v[2] = stream;
    ctr.v[3] = replication;
    const gpre::Philox4x32Counter r = gpre::philox4x32(ctr, key);
    unsigned int* dst = out + idx * 4;
    dst[0] = r.v[0];
    dst[1] = r.v[1];
    dst[2] = r.v[2];
    dst[3] = r.v[3];
  }
}

// Paths to device memory, layout out[step * n_paths + path] (coalesced), steps 0..n_steps with
// row 0 = s0. Expected to be memory-bound.
extern "C" __global__ void gbm_paths_store(unsigned long long seed, unsigned int stream,
                                           unsigned int replication, int n_paths, int n_steps,
                                           const float* __restrict__ drift_dt,
                                           const float* __restrict__ vol_sqrt_dt, float s0,
                                           int antithetic, float* __restrict__ out) {
  paths_store<float>(seed, stream, replication, n_paths, n_steps, drift_dt, vol_sqrt_dt, s0,
                     antithetic, out);
}

extern "C" __global__ void gbm_paths_store_f64(unsigned long long seed, unsigned int stream,
                                               unsigned int replication, int n_paths, int n_steps,
                                               const double* __restrict__ drift_dt,
                                               const double* __restrict__ vol_sqrt_dt, double s0,
                                               int antithetic, double* __restrict__ out) {
  paths_store<double>(seed, stream, replication, n_paths, n_steps, drift_dt, vol_sqrt_dt, s0,
                      antithetic, out);
}

// Terminal payoff in registers; per-tile float64 sums of the discounted payoff and its square.
// blockDim.x must be a multiple of 64; dynamic shared memory = 2 * blockDim.x * 8 bytes.
extern "C" __global__ void gbm_european_fused(unsigned long long seed, unsigned int stream,
                                              unsigned int replication, int n_paths, int n_steps,
                                              const float* __restrict__ drift_dt,
                                              const float* __restrict__ vol_sqrt_dt, float s0,
                                              float strike, int is_call, double discount,
                                              int antithetic, double* __restrict__ block_sum,
                                              double* __restrict__ block_sumsq) {
  european_fused<float>(seed, stream, replication, n_paths, n_steps, drift_dt, vol_sqrt_dt, s0,
                        strike, is_call, discount, antithetic, block_sum, block_sumsq);
}

extern "C" __global__ void gbm_european_fused_f64(unsigned long long seed, unsigned int stream,
                                                  unsigned int replication, int n_paths,
                                                  int n_steps, const double* __restrict__ drift_dt,
                                                  const double* __restrict__ vol_sqrt_dt, double s0,
                                                  double strike, int is_call, double discount,
                                                  int antithetic, double* __restrict__ block_sum,
                                                  double* __restrict__ block_sumsq) {
  european_fused<double>(seed, stream, replication, n_paths, n_steps, drift_dt, vol_sqrt_dt, s0,
                         strike, is_call, discount, antithetic, block_sum, block_sumsq);
}

// Arithmetic-average payoff over the steps flagged in fixing_mask, in registers (validated
// in M3 with the Asian pricer).
extern "C" __global__ void gbm_asian_fused(
    unsigned long long seed, unsigned int stream, unsigned int replication, int n_paths,
    int n_steps, const float* __restrict__ drift_dt, const float* __restrict__ vol_sqrt_dt,
    float s0, float strike, int is_call, const int* __restrict__ fixing_mask, float discount,
    int antithetic, double* __restrict__ block_sum, double* __restrict__ block_sumsq) {
  asian_fused<float>(seed, stream, replication, n_paths, n_steps, drift_dt, vol_sqrt_dt, s0, strike,
                     is_call, fixing_mask, static_cast<double>(discount), antithetic, block_sum,
                     block_sumsq);
}
