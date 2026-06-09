# kHeavyHash (Kaspa) — benchmark

Throughput of the kHeavyHash **OpenCL (AMD)** `search` kernel.

Two independent measurements are recorded and they agree:

1. the **`sources/benchmark` framework** (`bin/benchmark`), which times the bare
   kernel launch — no stratum, no share submission; and
2. the **end-to-end miner** (`bin/miner`) against a live pool, reading the
   steady-state dashboard hashrate.

> The kernel is the **correctness-only** implementation (one work-item per nonce,
> a full keccak-f[1600] per hash, the matrix re-read from global memory, no LDS
> caching or vectorisation). It is intentionally **not optimised** — these numbers
> are a baseline to tune against, not a peak.

## 1. `sources/benchmark` framework

The kHeavyHash benchmark lives at `sources/benchmark/amd/kheavyhash.cpp`
(`runAmdKHeavyHash`) and is wired into `runAmd()`. It generates the 64×64 nibble
matrix host-side (the same CPU reference the kernel is gated bit-identical
against), uploads matrix/header/target, then launches the `search` kernel in a
loop, timing each launch. The target is left all-zero so no work-item ever meets
it: the result buffer stays untouched and atomic contention never skews the
timing. Hashrate = (`blocks` × `threads`) nonces ÷ per-launch time.

Enable it in `sources/benchmark/config.json` (copied next to the binary) under
`amd.algorithms.kheavyhash` (`enabled: true`), then:

```bat
:: from the benchmark binary's directory
benchmark.exe config.json
```

The benchmark is **synchronous** (one `enqueue` + `finish` per launch, no
double-buffering), so the first ~30 launches measure the GPU ramping from idle
clocks; let it run ≥150 loops for a steady-state reading.

### Results

RX 9070 XT (RDNA4 / gfx1201), grid 8192 × 256, 200 loops, steady-state median
(first 40 ramp iterations dropped). All four kernels are gated **bit-identical**
by the OpenCL KAT (`opencl_kat_test.cpp`, run on the GPU via `unit_test.exe`).

| Kernel | Hashrate (steady median) | vs baseline | Optimization |
|---|---|---|---|
| `search`     | 393 MH/s | 1.00× | reference (one work-item/nonce, matrix re-read from global) |
| `search_lm1` | 414 MH/s | 1.05× | matrix staged in LDS (`__local uchar[4096]`) |
| `search_lm2` | 482 MH/s | 1.23× | + `v_dot4_u32_u8` matmul (`__builtin_amdgcn_udot4`) |
| `search_lm3` | 481 MH/s | 1.23× | + register-resident (unrolled rho/pi) keccak (no gain over lm2) |
| `search_lm4` | **573 MH/s** | **1.46×** | + powHash keccak **midstate** (per-job round-1 hoisted to LDS) |
| `search_lm5` | 574 MH/s | 1.46× | + heavy (2nd) keccak round-0 constant-parity hoist (ties lm4, within noise) |

`search_lm4` is the shipped kernel (`ResolverAmdKHeavyHash::buildSearch`). Numbers
are the median of two contention-free runs (569 / 577 MH/s for lm4); the other four
reproduce across runs within ~1%.

**lm5 ties lm4 (no reliable gain).** Validated over five back-to-back idle-GPU runs
(engine util <0.3% each): lm5 median 574.2 MH/s vs lm4 573.0, with lm5 consistently
a hair ahead — per-run deltas {−0.2, +0.2, +0.4, +0.4, +0.7}%, mean +0.31%. That is
*below the noise floor*: smaller than each kernel's own run-to-run spread (~0.7%,
lm4 570.5–574.2 / lm5 572.2–576.3). The round-0 hoist's ~16 saved XORs are real but
negligible — see "Next lever" below for why. lm4 stays the shipped kernel; lm5 is
retained in-tree as the recorded second-keccak experiment (like lm3, a tied negative
result). The baseline `search` reproduced tightly at 393.2 MH/s (392.4–395.0) across
the same five runs, confirming the measurement was clean.

#### RGA static analysis (gfx1201, offline)

| Kernel | VGPR | →waves/SIMD | global_load | `v_dot4` | LDS B | spills |
|---|---|---|---|---|---|---|
| `search`     | 129 | ~11 (VGPR-bound) | 22 | 0 | 8192 | 0 |
| `search_lm1` |  94 | 16 (max)         |  7 | 0 | 12288 | 0 |
| `search_lm2` |  84 | 16               |  7 | 32 | 12288 | 0 |
| `search_lm3` |  83 | 16               |  7 | 32 | 12288 | 0 |
| `search_lm4` |  90 | 16               |  7 | 32 | 12488 | 0 |
| `search_lm5` |  90 | 16               |  7 | 32 | 12488 | 0 |

#### What the numbers say

- **LDS staging alone (lm1) bought only +5%** despite dropping VGPR 129→94 and
  raising occupancy from ~11 to the full 16 waves/SIMD. The baseline was *not*
  occupancy-bound — it is compute-bound, so extra waves barely helped. (Same
  pattern as the progpow/ethash roofline findings, opposite conclusion: here the
  bottleneck is ALU, not memory latency.)
- **The `v_dot4` matmul (lm2) is the first real win, +22%.** gfx1201's OpenCL
  compiler exposes no `cl_khr_integer_dot_product`, but `__builtin_amdgcn_udot4`
  lowers to `v_dot4_u32_u8` (guarded by `__AMDGCN__`, bit-identical scalar fallback
  for POCL/CI). 4096 nibble-MACs/nonce → 1024 hardware dot ops.
- **The kernel is keccak-bound after the matmul** — but it is the keccak *work*
  that matters, not its unrolling: lm3 (unrolled rho/pi, register-resident) tied
  lm2, while **lm4 (keccak midstate) is the biggest single win, +19% over lm2**.
  In powHash only `state[9]` (nonce) varies per nonce; `prePowHash` + `timestamp`
  are fixed for the job, so the first keccak's round-1 (state init, the 32-byte
  absorb, and the theta) is computed **once per workgroup** (thread 0 → LDS) and
  each nonce just folds its nonce into 11 lanes and runs from round-1 ρ/π. The
  heavy (second) keccak depends on the matmul output and cannot be hoisted, so it
  stays full — which is why the ceiling is ~1.46×, not 2×.

## 2. End-to-end miner (corroboration)

```bat
miner.exe --algo kheavyhash --host kheavyhash.auto.nicehash.com --port 9200 ^
          --wallet <addr> --workername bench --password x --api_port 38080
```
Read the steady-state `Hashrate` column of the `HASHRATE` dashboard (printed every
~10 s), or query the HTTP API: `GET http://127.0.0.1:<api_port>/api/get_stats` →
`hs` is the per-device hashrate in h/s. Any kHeavyHash pool works (unmineable /
NiceHash / 2miners); the hashrate is the kernel's throughput, independent of pool.

| GPU | Arch | Hashrate | Notes |
|---|---|---|---|
| AMD Radeon RX 9070 XT | RDNA4 (gfx1201) | **~389.8 MH/s** | two consecutive readings: 389.87, 389.80 MH/s |

The miner double-buffers (two command queues overlapping), so its sustained
number lands just under the framework's bare-launch peak — the two agree within
~2%, which is the expected gap between an overlapped pipeline and an isolated
`finish`-per-launch micro-benchmark.

Host: Windows 11; binary cross-compiled from Linux via `docker/Dockerfile.windows-cross`
(llvm-mingw + vcpkg `x64-mingw-static`); runtime OpenCL ICD from the AMD Radeon driver.
Correctness on the same run was confirmed end-to-end (pool-accepted shares).

## Caveats / follow-ups

- The optimisation pass (lm1→lm4) took the kernel **392 → 573 MH/s (+46%)**. Two
  levers carried it: the `v_dot4` matmul (lm2, +22%) and the powHash keccak
  midstate (lm4, +19% on top). LDS staging (lm1) and keccak unrolling (lm3) were
  each marginal. Measured the same way as the baseline, so directly comparable.
- **Heavy (second) keccak lever — TESTED (lm5), marginal.** lm4 hoisted the
  *first* keccak's per-job-constant round-1; the second keccak (`kHeavyHash`, over
  the matmul output) is fully nonce-dependent and stays at 24 rounds — it is now
  the largest single cost. lm5 applied idea (a): the heavy keccak absorbs only 32
  bytes into the constant `HEAVY_INITIAL_STATE` (lanes 4..24 fixed), so its round-0
  theta column-parity `c4` is fully constant and `c0..c3` collapse to
  `input_lane ^ const` — trimming round-0 parity reduction from 20 XORs to 4. **It
  ties lm4 within noise.** Why the lm4 trick doesn't transfer: the nonce touches a
  *single* column of powHash (so the whole round-0 theta hoisted), but the matmul
  product fills lanes 0..3 — one element in *four* of the five columns. So every
  round-0 theta D-value depends on the input, nothing hoists past round-0 theta,
  and chi's nonlinearity blocks pushing the input any further. ~16 saved XORs out
  of ~24 rounds is sub-1%. **This puts the heavy keccak at its floor and confirms
  the ~1.45× ceiling** — the remaining cost is two essentially-irreducible 24-round
  permutations plus the matmul, all at max occupancy (16 waves/SIMD) with no spills.
  RGA shows lm5 identical to lm4 in VGPR/occupancy/`v_dot4`; only ISA size grows
  (12204 vs 11160 B) from the unrolled round-0, for no throughput. Idea (b)
  (2-nonce/thread to hide 64-bit keccak latency) is not worth pursuing: the lm1
  evidence (occupancy 11→16 bought only +5%) shows the kernel is ALU-throughput
  bound, not latency bound, so adding per-thread ILP at the cost of occupancy/VGPR
  would not help. lm5 is retained in-tree as the recorded experiment; lm4 ships.
- The ~1.3–1.7 GH/s figure cited for other miners was not reproduced and is not a
  like-for-like comparison (different nonce-batching / pipeline). Treat ~573 MH/s
  as the verified single-work-item-per-nonce result for the current algorithm shape.
- **Benchmark hygiene:** these numbers require an otherwise-idle GPU. A concurrent
  workload on the same card (e.g. another miner) halves and de-orders the results,
  and a load that resumes mid-run corrupts the later kernels — run on an idle GPU
  and sanity-check that the `search` baseline lands near ~393 MH/s and `lm2 > lm1`.
