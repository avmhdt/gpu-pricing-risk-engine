# Random numbers

## Philox4x32-10 (FR-02)

Philox4x32-10 (Salmon, Moraes, Dror and Shaw, SC '11) is a keyed bijection on a 128-bit counter.
Each of the ten rounds multiplies two of the four 32-bit words by fixed constants and mixes the
high and low halves of the 64-bit products with the other words and the key; the key is bumped
by the Weyl constants between rounds:

| Constant | Value |
|---|---|
| M0, M1 (multipliers) | `0xD2511F53`, `0xCD9E8D57` |
| W0, W1 (key bumps) | `0x9E3779B9`, `0xBB67AE85` |

Round: `(hi0, lo0) = M0 * c0`, `(hi1, lo1) = M1 * c2`, then
`c = (hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0)`.

Three implementations must agree bit for bit:

| Implementation | File | Checked by |
|---|---|---|
| NumPy reference (uint64 arithmetic) | `src/gpre/rng/philox.py` | `tests/unit/test_philox.py` |
| Host C++20 (same header as CUDA) | `src/gpre/cuda/kernels/philox4x32.cuh` | `cpp/tests/test_philox.cpp` |
| CUDA device code (NVCC and NVRTC) | `src/gpre/cuda/kernels/philox4x32.cuh`, `paths.cu` | `tests/gpu/test_rng_invariance.py` (M2) |
| JAX (Should) | `src/gpre/rng/` (M2) | — |

All of them read the Random123 known-answer vectors in `tests/fixtures/philox_kat.json`
(Philox4x32-7 and -10, three vectors each).

## Key and counter layout (FR-03)

- **Key:** the 64-bit master seed `GPRE_SEED` split as `(k0, k1) = (low word, high word)`.
- **Counter:** `(c0, c1, c2, c3) = (draw block j, path p, stream s, replication r)`. One Philox
  call gives the four normals for steps `4j … 4j+3` of path `p`.

Because the output depends only on `(seed, s, r, p, step)`, it does not change with the grid
size, block size, thread count or chunking. `philox_uniforms` in `paths.cu` writes the raw words
in this layout for the bit-exactness tests.

The uniform and normal conversions, antithetic pairing and the fixed table of stream ids per
(underlying, purpose) are defined with the path generators in M2.
