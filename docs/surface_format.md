# Surface files: vsm.surface.v1

vol-surface-modeling owns the format. Its rules are in that repo's `docs/export-format.md` and its
JSON Schema in `schemas/vsm.surface.v1.schema.json`, vendored here as
`src/gpre/data/schemas/vsm.surface.v1.schema.json`. This engine reads exactly that format and
writes it for the files it produces itself. A needed change is requested in vol-surface-modeling:
additive optional fields first, breaking changes only as vsm.surface.v2.

## Reading

`gpre.data.surface_io.read_surface` validates every file against the vendored schema, then parses
the subset below into a pydantic model. A file with `schema` other than `vsm.surface.v1` is
rejected.

| Field | Use here |
|---|---|
| `schema` | Must equal `vsm.surface.v1` |
| `underlying`, `venue` | Matched against `configs/universe.yaml` (from M5) |
| `as_of`, `as_of_ns` | Valuation time; t is measured from here (ACT/365 seconds) |
| `spot.value` | Initial underlying level |
| `conventions.*` | Recorded; units are converted only in ingest modules (DQ-07) |
| `canonical` | `ssvi`: the continuum is SSVI. `svi`: allowed only with `ssvi = null` |
| `ssvi.t_nodes`, `ssvi.theta_nodes`, `rho`, `phi`, `eta`, `gamma`, `lambda` | SSVI continuum |
| `expiries[].t`, `forward`, `discount_factor`, `basis_annualized`, `dividend_yield` | Forwards and discounting per expiry |
| `expiries[].svi_raw` | Raw SVI slices (rule 3) |
| `expiries[].slice_source`, `fit`, `forward_method`, `forward_se`, `k_range_quoted`, `arbitrage` | Logged, not trusted: DQ-05 re-checks |
| `heston`, `sabr`, `local_vol` | Not used for pricing (the engine computes Dupire local vol itself) |
| `provenance.producer` | Recorded in run metadata |

## Evaluation rules

1. **Continuum (`canonical = ssvi`):** w(k, t) is SSVI with θ(t) linear between nodes, θ₁·t/t₁
   below the first node and θₙ·t/tₙ beyond the last. `SSVI.w` in `gpre.surface.ssvi`.
2. **Forward:** ln F linear in t between expiries; beyond the ends F(t) = S·exp(b·t) with the
   nearest end's `basis_annualized`. `Surface.forward`.
3. **Slices (`canonical = svi`):** linear in w at fixed k between expiries. Beyond the ends the
   engine scales the end slice by t/t_end, as rule 1 does for θ (the producer does not define
   the ends for slices).

## DQ-05

`check_arbitrage` evaluates every maturity in the file (expiries and SSVI nodes) on k from −1 to
1 in steps of 0.01:

- calendar: w(k, Tᵢ₊₁) − w(k, Tᵢ) ≥ −1e−12;
- butterfly: Durrleman's g(k) ≥ −1e−10 on each slice (the exact raw-SVI form of an SSVI slice);
- for information, the largest |1 − ∫p| of the implied density over all maturities.

A failing file is rejected with a report naming the maturity and k of the worst violation.

## Files written here

The D-05 synthetic surface (`gpre surface synthetic`, fixture `tests/fixtures/surface_ssvi.json`)
has θ(T) = 0.04·T, ρ = −0.7 and power-law φ with η = 1.0, γ = 0.5, nodes at the book tenors
(1M, 3M, 6M, 1Y), S = 100, r = 3%, q = 1%. It sets `canonical = ssvi`, one `expiries[]` entry
per node whose `svi_raw` is the exact raw-SVI form of that SSVI slice (`slice_source =
ssvi_fallback`, `fit.n_quotes = 0`), `heston`, `sabr` and `local_vol` empty, and
`provenance.producer = synthetic`. Field choices forced by the schema's enums are listed in
`docs/decisions.md`.
