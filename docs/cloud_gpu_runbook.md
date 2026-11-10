# Cloud GPU runbook (Hyperstack, D-16)

The owner creates the Hyperstack account, loads the prepaid balance (at or below USD 50) and
registers an SSH key (P-17). The owner launches and ends every VM. The build agent never rents or
pays for anything.

## Before the first session (owner, once)

1. Create the account at hyperstack.cloud and complete the identity check. You are in Brazil at
   the moment; if sign-up or payment asks for a country, use your real one.
2. Load a prepaid balance **at or below USD 50**. On-demand usage is prepaid and billed per minute;
   resources are stopped when the balance reaches zero, so the balance is the hard cap.
3. Add your SSH public key (Console → SSH keys). If the agent should drive the session from this
   PC's WSL, the matching private key must be in `~/.ssh` there; otherwise you run the commands
   below yourself.
4. Put `GPRE_CLOUD_PROVIDER=hyperstack` in this repo's `.env`. No API key is needed.

## First session: M2 GPU exit checks

**Expected length:** about 60 to 90 minutes on one RTX A4000.
**Expected cost:** USD 0.15 to 0.25. If the VM is deleted at the end, nothing else bills.

### 1. Launch (owner, in the console)

| Setting | Value |
|---|---|
| Region | `NORWAY-1` (where the A4000 flavor is listed) |
| Flavor | `n3-RTX-A4000x1` (1 × RTX A4000 16 GB, 4 vCPU, 21 GB RAM, 100 GB root) at $0.15/h |
| Image | The newest **Ubuntu 22.04 … CUDA 12.x** image in the list (for example "Ubuntu Server 22.04 LTS R535 CUDA 12.2"); note its exact name |
| SSH key | The key from step 3 above |
| Public IP | Enabled |

Write down the launch time (UTC). Locally, `uv run gpre spend check` must say `ok` first.

### 2. Set up (about 10 minutes)

```bash
ssh ubuntu@<public-ip>
curl -LsSf https://raw.githubusercontent.com/avmhdt/gpu-pricing-risk-engine/m2-rng-paths-european/scripts/cloud_setup.sh -o cloud_setup.sh
bash cloud_setup.sh m2-rng-paths-european
```

The script prints the GPU, driver and CUDA version, installs `uv`, clones the repo, installs the
`dev` and `gpu` groups and ends with `gpre env --require cuda` and `--require gpu`. Both must
pass. If either fails, stop here, copy the error, and go to step 6. Nothing else is worth paying
for without a working GPU stack.

### 3. Nsight Compute counter check (the first 10 minutes after setup)

```bash
cd ~/gpre && bash scripts/ncu_check.sh
```

It profiles one fused kernel, first as the user and then with `sudo`, and prints a verdict:

- **Counters available:** the roofline in M6 uses measured metrics.
- **Blocked even with sudo** (`ERR_NVGPUCTRPERM`): the analytic roofline fallback applies (FR-28,
  R-02). This does not block M2 to M5. The result is recorded either way.

### 4. M2 GPU work (about 30 to 40 minutes)

```bash
cd ~/gpre
JAX_ENABLE_X64=1 uv run pytest -m gpu -q                     # AC-W-06 CUDA part, kernel agreement
uv run gpre validate --suite rng                              # Philox on NumPy and JAX
JAX_ENABLE_X64=1 uv run gpre validate --suite european --backends numpy,jax,cuda   # AC-W-01
uv run gpre bench throughput --backends cuda,jax --paths 4194304 --steps 252      # first numbers
uv run gpre env                                               # hardware stamp for this instance
```

The throughput numbers are preliminary: published benchmarks follow the shared-PC rules and the
full table is M6.

### 5. Copy results back (from this PC, in WSL)

```bash
cd /mnt/d/trading-strategy-brainstorm/gpu-pricing-risk-engine
rsync -av ubuntu@<public-ip>:gpre/results/ ./results/
```

Only small JSON and CSV files come back. The agent reviews and commits them; nothing is pushed
from the VM, so no GitHub credentials ever go on it.

### 6. End the session: delete it, never just shut it off

1. Note the stop time and read the session's cost on the billing page.
2. Log it locally before ending:
   `uv run gpre spend add --instance-type n3-RTX-A4000x1 --gpu-model "NVIDIA RTX A4000" --start <UTC> --stop <UTC> --cost-usd <cost> --end-state deleted`
3. In the console, **Delete** the VM. A **SHUTOFF** VM keeps billing the full rate. Hibernate
   bills only the saved disk and a kept IP, but Hyperstack limits hibernation to some flavors
   and does not say whether the A4000 is one, so delete is the default. `cloud_setup.sh`
   rebuilds everything in about 10 minutes.
4. Check that the VM list is empty and the balance has stopped moving.

## Spend plan for the whole project

| Milestone | GPU work | A4000 hours | Cost |
|---|---|---|---|
| M2 | Setup, ncu check, RNG and European validation, first throughput | 1–1.5 | $0.15–0.25 |
| M3 | Asian, barrier, LSM, Heston, local-vol validation on all backends | 2–3 | $0.30–0.45 |
| M4 | Greeks and AAD validation, first AAD cost | 1–2 | $0.15–0.30 |
| M5 | Book revaluation, VaR, 2019–2025 backtest (budget ≤ 3 h) | 3–5 | $0.45–0.75 |
| M6 | Ceilings, ncu profiles, speedup table, QMC curves, AAD cost, precision | 4–6 | $0.60–0.90 |
| M7 | Final `make all` GPU parts | 2–3 | $0.30–0.45 |
| Optional | One A100 session (`n3-A100x1`, CANADA-1, $1.35/h) for the FP64 side of AC-H-06 | 1–2 | $1.35–2.70 |
| **Total** | | **13–22** | **about $3–6** |

Work pauses at 80% of the cap (USD 40) until the owner confirms (`gpre spend check` enforces it),
and `gpre spend add` refuses any row that would pass USD 50. Allowing for per-minute billing,
setup time and a few failed sessions, real spend should stay well under USD 10.

## Provider research behind D-16 (2026-11-09)

| | Hyperstack (chosen) | AWS g6.xlarge | Vast.ai VM offers |
|---|---|---|---|
| GPU and price | RTX A4000 $0.15/h; A100 $1.35/h | L4 about $0.80/h | A4000 about $0.12/h (marketplace) |
| Billing | Per minute, prepaid | Per second, 60 s minimum | Per second, prepaid |
| Stopped instance | SHUTOFF bills in full; hibernate bills disk and IP only | Compute stops; disk bills | Disk bills |
| Hard cap | Prepaid balance | None (budget alerts only); GPU quota starts at 0 | Prepaid, auto-billing off |
| Nsight counters | Unconfirmed (checked in step 3) | Reported working | Unconfirmed |

Lambda was ruled out because its staff said Nsight Compute is not supported. RunPod pods are
unprivileged containers. No provider names the Cayman Islands or Brazil; their terms exclude only
sanctioned countries.
