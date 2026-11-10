# Decisions

Owner decisions that apply to this project, copied from the portfolio's `DECISIONS.md` at M1
(2026-11-09), followed by the choices the build agent made where the spec left room (SPEC §16:
"choose the simplest option consistent with it, record the choice here").

## Owner decisions (from `DECISIONS.md`, recorded 2026-10-31 unless dated)

| Item | Decision |
|---|---|
| Scope | Project 14 of 15, started in the fourth batch on 2026-11-09 |
| Merging | Feature branch per milestone, PR, squash auto-merge with `gh pr merge --auto --squash` as `avmhdt` once CI is green; branch protection on once CI exists |
| Country | Cayman Islands (`OWNER_COUNTRY=KY`) |
| P-10, D-02 cloud GPU budget | USD 50 cap, approved. Spend log in `results/cloud_spend.csv`; GPU work pauses at 80% of the cap until the owner confirms; the instance is shut down after every session |
| D-16 cloud GPU provider | **Open.** The agent researches hourly providers that serve the Cayman Islands and brings two or three options before M2; the owner picks |
| P-17 provider account | Owner creates the account, enters payment details and sets a provider-side billing limit at or below the cap. Agents never rent or pay |
| D-01 compute platform | Cloud GPU; this PC has no NVIDIA GPU, so local work is CPU-only. No NVIDIA driver or CUDA toolkit is installed locally |
| D-15 price source | Sharadar's direct API (SEP stocks, SFP for SPY), key in `SHARADAR_API_KEY`, sent only in the `x-api-key` header. Fallback and cross-check: the owner's local Stooq snapshot |
| Sharadar licence | Personal Use License; Sharadar confirmed in writing (2026-10-31) that public code, aggregate results and job-application use are allowed. Publish aggregates only, never security-level data |
| Data placement | Data under `DATA_ROOT` and build trees under `~/build/gpu-pricing-risk-engine`, both on WSL ext4; never on `C:`, never in git |
| Benchmarks | Published GPU and CPU benchmark runs wait until no multi-week capture runs on this PC (after about 2026-11-29) |
| Shared PC limits | Downloads at most 10 MB/s; heavy runs at most 8 threads at `nice 10`; stop if free memory falls below 30 GB; keep `D:` above 150 GB free; never restart WSL |
| Attribution | No AI or assistant attribution anywhere in the repository |
| Other spec defaults | Every other decision in SPEC §3 with a default stands as written (D-03 to D-14, D-17 MIT) |

## Agent choices

### M1 (2026-11-09)

- **Pinned versions.** `uv.lock` resolves Python 3.12 with JAX 0.12.0.1 (the `cuda12` extra still
  exists), CuPy `cupy-cuda12x` 14.2 in the `gpu` group, NumPy 2.5, SciPy 1.18, Polars 2.1,
  QuantLib 1.44 (dev only). C++: GCC 13, CMake ≥ 3.28 with CPM.cmake 0.43.2, GoogleTest 1.18.0,
  nlohmann/json 3.12.0. CI's CUDA compile check uses `nvidia/cuda:12.9.1-devel-ubuntu24.04` and
  compiles for `sm_70`, `sm_80` and `sm_89`. The working GPU combination (driver, CUDA, JAX,
  CuPy) is recorded here at M2 on the chosen instance.
- **Philox key.** The 64-bit seed is split into the key as (low word, high word), identically in
  `gpre.rng.philox.seed_to_key` and `philox_key_from_seed` in `philox4x32.cuh`.
- **Sharadar row cap.** The table API answers in CSV, newest row first, and cut an unkeyed AAPL
  2012–2025 query at 1,039 rows on 2026-11-09. The ingest therefore asks for one symbol-year at a
  time (≤ 253 rows) and treats an answer at the 1,000-row cap as an error.
- **Sharadar adjustment.** `close` is split-adjusted and feeds `log_ret`, since options are
  written on the price rather than the total return. `closeadj` (split and dividend) is kept as
  `close_adj` and `closeunadj` as `close_unadj`. Facts as checked by equity-factor-pipeline on
  the Full History Bundle.
- **Key checks.** AAPL, MSFT and SPY answer without a valid key, so the ingest first checks the
  key with GME (SEP) and XLE (SFP).
- **Stooq adjustment.** Stooq closes are treated as split-adjusted; DQ-04 fails on any move that
  matches a split ratio. regime-portfolio-construction found that Stooq misses many
  distributions, so Stooq is not total-return data and is used only as fallback.
- **M1 price source.** `SHARADAR_API_KEY` is not yet in this repo's `.env`, so the M1 data smoke
  test ran on the Stooq fallback (allowed by AC-W-13). The Sharadar path is tested with mocked
  HTTP and runs for real once the owner adds the key.
- **Universe (D-14, provisional).** SPY plus 49 large caps whose tickers did not change over
  2012–2025 and which had no spin-off or reverse split big enough to break a continuous history
  (GE, T and MMM were left out for that reason). The list is in `configs/universe.yaml` and is
  confirmed against Sharadar at M5. It is chosen with hindsight, so it carries survivorship
  bias (DQ-11); acceptable for VaR model testing and stated in the README's limitations.
- **FRED.** Without `FRED_API_KEY` the public `fredgraph.csv` endpoint is used. With a key, FRED
  v1 accepts it only as the `api_key` query parameter, so URLs are redacted before they are
  logged or written to the raw manifest. Trailing missing values beyond the 3-day fill are
  dropped (no later value exists to bound them); interior gaps over 3 business days fail.
- **Raw immutability (DQ-09).** A raw file that already exists for the same day is reused
  instead of downloaded again. Stooq files are read in place on `E:` and recorded in the
  manifest by absolute path and SHA-256.
- **`gpre data check --strict`.** Strict mode fails on warnings and on skipped rules (an input
  not ingested), so a strict pass means every M1 rule ran and passed. `results/data_check.json`
  holds counts and dates only, never prices or VIX values.
- **Synthetic surface fields.** vsm.surface.v1 has closed enums designed for crypto surfaces.
  The D-05 synthetic equity surface uses `settlement = cash_usd`, `coin_rate = implied` (its
  `discount_factor` is exp(−rT)), `forward_method = future` (a theoretical forward
  S·exp((r−q)T)) and `arbitrage.grid` = the DQ-05 k grid (−1 to 1, 201 points). Expiry times are
  whole seconds, so t is exact on ACT/365 seconds.
- **Run manifests** are committed under `results/manifests/<run_id>.json`; per-run JSON logs stay
  local in `results/logs/` (git-ignored).
- **Development venv.** Kept on WSL ext4 via `UV_PROJECT_ENVIRONMENT`
  (`~/build/gpu-pricing-risk-engine/venv`), since `/mnt/d` is slow from WSL.
