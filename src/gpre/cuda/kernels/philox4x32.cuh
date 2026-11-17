// Philox4x32-10 counter-based generator (Salmon et al. 2011), shared by the CUDA kernels
// (NVCC and NVRTC) and the C++ baseline (host C++20). Must agree bit for bit with the NumPy
// reference in gpre/rng/philox.py; the Random123 known-answer vectors in
// tests/fixtures/philox_kat.json are checked by every implementation.
//
// No standard headers: NVRTC compiles this without a host include path.
#ifndef GPRE_PHILOX4X32_CUH_
#define GPRE_PHILOX4X32_CUH_

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define GPRE_HD __host__ __device__ __forceinline__
#else
#define GPRE_HD inline
#endif

namespace gpre {

struct Philox4x32Counter {
  unsigned int v[4];
};

struct Philox4x32Key {
  unsigned int v[2];
};

constexpr unsigned int kPhiloxM0 = 0xD2511F53U;
constexpr unsigned int kPhiloxM1 = 0xCD9E8D57U;
constexpr unsigned int kPhiloxW0 = 0x9E3779B9U;
constexpr unsigned int kPhiloxW1 = 0xBB67AE85U;
constexpr int kPhiloxRounds = 10;

// High and low 32-bit words of a 32x32-bit product.
GPRE_HD void philox_mulhilo(unsigned int a, unsigned int b, unsigned int* hi, unsigned int* lo) {
#if defined(__CUDA_ARCH__)
  *lo = a * b;
  *hi = __umulhi(a, b);
#else
  const unsigned long long product =
      static_cast<unsigned long long>(a) * static_cast<unsigned long long>(b);
  *lo = static_cast<unsigned int>(product);
  *hi = static_cast<unsigned int>(product >> 32U);
#endif
}

GPRE_HD Philox4x32Counter philox_round(Philox4x32Counter ctr, Philox4x32Key key) {
  unsigned int hi0 = 0;
  unsigned int lo0 = 0;
  unsigned int hi1 = 0;
  unsigned int lo1 = 0;
  philox_mulhilo(kPhiloxM0, ctr.v[0], &hi0, &lo0);
  philox_mulhilo(kPhiloxM1, ctr.v[2], &hi1, &lo1);
  Philox4x32Counter out;
  out.v[0] = hi1 ^ ctr.v[1] ^ key.v[0];
  out.v[1] = lo1;
  out.v[2] = hi0 ^ ctr.v[3] ^ key.v[1];
  out.v[3] = lo0;
  return out;
}

// Philox4x32 with `rounds` rounds (10 = Philox4x32-10, the generator used everywhere here).
GPRE_HD Philox4x32Counter philox4x32(Philox4x32Counter ctr, Philox4x32Key key,
                                     int rounds = kPhiloxRounds) {
  for (int r = 0; r < rounds; ++r) {
    if (r > 0) {
      key.v[0] += kPhiloxW0;
      key.v[1] += kPhiloxW1;
    }
    ctr = philox_round(ctr, key);
  }
  return ctr;
}

// Key from a 64-bit seed: (low word, high word), as gpre.rng.philox.seed_to_key.
GPRE_HD Philox4x32Key philox_key_from_seed(unsigned long long seed) {
  Philox4x32Key key;
  key.v[0] = static_cast<unsigned int>(seed & 0xFFFFFFFFULL);
  key.v[1] = static_cast<unsigned int>(seed >> 32U);
  return key;
}

// Uniforms strictly inside (0, 1), exact in their precision (docs/rng.md, FR-03).
// float32: one word, u = ((x >> 9) + 0.5) * 2^-23  (four uniforms per Philox call).
GPRE_HD float philox_uniform_f32(unsigned int x) {
  return (static_cast<float>(x >> 9U) + 0.5F) * 1.1920928955078125e-07F;
}

// float64: two words, u = ((a << 20 | b >> 12) + 0.5) * 2^-52  (two uniforms per call).
GPRE_HD double philox_uniform_f64(unsigned int a, unsigned int b) {
  const unsigned long long k = (static_cast<unsigned long long>(a) << 20U) | (b >> 12U);
  return (static_cast<double>(k) + 0.5) * 2.220446049250313080847263336181640625e-16;
}

}  // namespace gpre

#endif  // GPRE_PHILOX4X32_CUH_
