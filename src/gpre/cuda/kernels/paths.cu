// Path-generation kernels, compiled at run time by CuPy RawModule (NVRTC) and checked by nvcc
// in CI (cuda-compile job). M1 holds the Philox uniform kernel used by the bit-exactness tests;
// gbm_paths_store, gbm_european_fused, gbm_asian_fused, bandwidth_probe and fma_probe follow in
// M2 and M6 with the signatures in SPEC §10.
#include "philox4x32.cuh"

// Raw Philox output for the stream layout of FR-03:
//   key = seed, counter = (draw block j, path p, stream, replication),
//   out[((p * n_blocks_per_path) + j) * 4 + i] = word i of Philox(counter, key).
// One thread per (path, block) pair; the output does not depend on the launch configuration.
extern "C" __global__ void philox_uniforms(unsigned long long seed, unsigned int stream,
                                           unsigned int replication, int n_paths,
                                           int n_blocks_per_path, unsigned int* __restrict__ out) {
  const long long total = static_cast<long long>(n_paths) * n_blocks_per_path;
  const gpre::Philox4x32Key key = gpre::philox_key_from_seed(seed);
  for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<long long>(gridDim.x) * blockDim.x) {
    const unsigned int path = static_cast<unsigned int>(idx / n_blocks_per_path);
    const unsigned int block = static_cast<unsigned int>(idx % n_blocks_per_path);
    gpre::Philox4x32Counter ctr;
    ctr.v[0] = block;
    ctr.v[1] = path;
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
