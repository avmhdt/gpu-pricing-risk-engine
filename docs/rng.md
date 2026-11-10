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
- **Counter:** `(c0, c1, c2, c3) = (draw block j, path p, stream s, replication r)`.
- **float32:** one Philox call gives the uniforms of steps `4j … 4j+3` of path `p`, one word
  each: `u = ((x >> 9) + 0.5) · 2^-23`.
- **float64:** one call gives steps `2j, 2j+1`, each from two words `(a, b)`:
  `u = ((a << 20 | b >> 12) + 0.5) · 2^-52`.
- **Normals** by inverse CDF: SciPy `ndtri` (NumPy reference), `jax.scipy.special.ndtri`, CUDA
  `normcdfinvf` / `normcdfinv`. Box–Muller is not used, because it would mix Sobol dimensions.
- **Antithetic:** path `p` uses Philox path `⌊p/2⌋` and flips the sign of the normals for odd `p`.

Because the output depends only on `(seed, s, r, p, step)`, it does not change with the grid
size, block size, thread count or chunking. Lengthening a path does not change its earlier steps.

### Why 23 bits for float32

The spec's float32 form `((x >> 8) + 0.5) · 2^-24` is exact in float64 but not in float32: for
the largest words, `k + 0.5` needs 25 significant bits, so the value rounds to exactly 1.0 and the
inverse CDF returns +∞, about once in 2^24 draws. A 4M-path × 252-step run would hit it about 60
times. With 23 bits every uniform is exact in float32 and lies in `[2^-24, 1 − 2^-24]`, so
`|z| ≤ 5.29`, which is the float32 tail bound the spec's own unit tests expect. The float64 form
uses 52 bits for the same reason.

### Uniforms in each implementation

| Implementation | Function |
|---|---|
| NumPy | `gpre.rng.streams.words_to_uniforms`, `philox_uniforms`, `philox_normals` |
| JAX | `gpre.rng.philox_jax.block_uniforms` (uint32 arithmetic only, 16-bit limbs for the products) |
| CUDA, C++ | `philox_uniform_f32`, `philox_uniform_f64` in `philox4x32.cuh` |

Words are bit-identical across all of them. Uniforms are bit-identical too, being exact
functions of the words. Normals agree to within the accuracy of each inverse-CDF routine.

## Stream ids

`s = underlying_index × 16 + purpose` (`gpre.rng.streams.stream_id`):

| Purpose | Code | Use |
|---|---|---|
| `paths` | 0 | Spot Brownian drivers |
| `variance` | 1 | Second Heston factor |
| `lsm_regression` | 2 | Independent LSM regression paths |
| `scenarios` | 3 | Monte Carlo VaR scenarios |
| `validation` | 4 | One stream per validation case (case index in place of the underlying) |

Positions on one underlying share stream `(i, paths)`, which gives common random numbers across
positions and scenarios (FR-18).

## Sobol and the Brownian bridge (FR-04)

- `scipy.stats.qmc.Sobol` with Joe–Kuo directions and Owen scrambling, sample sizes `2^m` only.
- Dimension = number of steps (× factors from M3). Replication `r` uses a scrambling seeded from
  `(seed, stream, r)`.
- Points are drawn in chunks from one engine, so the result does not depend on the chunk size.
- `sobol-bb` maps the coordinates through the bisection Brownian bridge (`gpre.rng.bridge`), so
  coordinate 1 drives W(T), coordinate 2 the midpoint, and so on. `sobol` uses time-ordered
  increments.
- In `price_mc`, `n_paths` is the total over R replications (R = 32 by default) of `n_paths / R`
  points each. The estimate is the mean of the replication means; the SE is their standard
  deviation over √R.

## Reductions in the fused kernels (NFR-02)

The fused kernels write one float64 (sum, sum of squares) pair per tile of 64 consecutive paths,
reduced in a fixed tree order inside the tile. Tile results are therefore identical for any
block size (a multiple of 64) and any grid size. The host merges tiles in index order with the
exact parallel variance formula (`gpre.cuda.launch.merge_tiles`). No floating-point atomics.
