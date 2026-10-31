# GPU-accelerated pricing and risk engine

> Monte Carlo pricing, Greeks and portfolio risk on a GPU, validated against closed forms and benchmarked against CPU.

**Status:** planning

| | |
|---|---|
| Track | Quant Developer |
| Estimated effort | 4–6 weeks |
| Stack | CUDA C++ or JAX, CuPy, Triton; NumPy and C++ baselines |
| Data cost | Free |
| Strongest for | Bank strats, risk technology, pricing libraries |

## Requirements

- [ ] Monte Carlo pricers: European (for validation), Asian, barrier, and American via Longstaff–Schwartz.
- [ ] Path generation under GBM, Heston and local vol (from [vol-surface-modeling](https://github.com/avmhdt/vol-surface-modeling)).
- [ ] Variance reduction: antithetic variates, control variates, Sobol quasi-random numbers with a Brownian bridge.
- [ ] Greeks by pathwise and likelihood-ratio methods, plus AAD (JAX autodiff or a hand-written adjoint).
- [ ] Portfolio VaR and Expected Shortfall, historical and Monte Carlo, with full revaluation.
- [ ] CPU baselines in NumPy and multithreaded C++.

## Data

Raw and licensed data live in `data/`, which is git-ignored. Only download scripts and small synthetic fixtures are committed.

| Dataset | Cost | What it gives you |
|---|---|---|
| Synthetic market parameters | Free | Most of the work runs on generated inputs. |
| [FRED Treasury yields](https://fred.stlouisfed.org) | Free | Discount curves. |
| [Your surfaces from [vol-surface-modeling](https://github.com/avmhdt/vol-surface-modeling), or Deribit](https://docs.deribit.com) | Free | Real volatility inputs. |
| [Stooq daily history](https://stooq.com/db/h/) | Free | Historical scenarios for VaR. |
| [Google Colab or cloud GPUs](https://colab.research.google.com) | Free tier | GPU access if you don’t own one. |

## Success criteria

### Works (correct and reproducible)

- [ ] Prices converge to closed forms (Black–Scholes, geometric Asian, barrier) within three standard errors.
- [ ] Greeks match finite differences, and AAD Greeks match bump-and-revalue.
- [ ] A VaR backtest whose exception counts pass the Kupiec test.

### Impresses hirers

- [ ] A speedup table against CPU baselines, with roofline analysis and achieved memory bandwidth.
- [ ] Error-vs-time curves showing where quasi-Monte Carlo beats plain Monte Carlo.
- [ ] AAD computing all Greeks for a small multiple of one pricing run.
- [ ] Full revaluation of a 10,000-position book in seconds.

## Questions this project must answer

- Is your kernel compute-bound or memory-bound?
- Why does QMC need a Brownian bridge?
- How do you keep random numbers reproducible across GPU threads?
