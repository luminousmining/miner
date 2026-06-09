# kHeavyHash — AMD OpenCL Optimisation Work

## Overview

This document covers the OpenCL kernel optimisations developed for kHeavyHash
(Kaspa) on AMD GPUs. Six variants — `kHeavyHash_lm0` (the reference) through
`kHeavyHash_lm5` — have been implemented, each adding one lever on top of the
previous one. `kHeavyHash_lm4` is the shipped kernel.

Unlike Ethash/KAWPOW, kHeavyHash is **not memory-hard**: there is no DAG. The only
per-job state is the 64×64 nibble matrix (generated host-side from the pre-pow
header), the 32-byte header and the 32-byte little-endian target. Each nonce runs
the full `powHash → heavyHash (matmul) → kHeavyHash` pipeline, so the kernel is
**ALU-bound**, not latency- or bandwidth-bound — the opposite regime to the
KAWPOW/Ethash roofline work.

Benchmark entry point: [sources/benchmark/amd/kheavyhash.cpp](../../sources/benchmark/amd/kheavyhash.cpp)
Kernel source (all variants in one file): [sources/algo/kheavyhash/opencl/kheavyhash.cl](../../sources/algo/kheavyhash/opencl/kheavyhash.cl)
Correctness gate: [sources/algo/kheavyhash/opencl/tests/opencl_kat_test.cpp](../../sources/algo/kheavyhash/opencl/tests/opencl_kat_test.cpp)

---

## Benchmark Setup

Each kernel is compiled at runtime via `KernelGeneratorOpenCL`, which injects
defines before building the OpenCL source, then launched in a timed loop. The
target is left all-zero so no work-item ever "meets" it: the result buffer stays
untouched and atomic contention never skews the timing. Hashrate =
(`blocks × threads`) nonces ÷ per-launch time.

- **Group size (`threads`)**: 256 work-items
- **Worker group count (`blocks`)**: 8192
- **Work-item collaboration**: none — **one work-item per nonce** (kHeavyHash has
  no lane-parallel structure; each nonce is independent)
- **Loop**: 200 iterations (drop the first ~40 cold/ramp iterations)

Key define injected at compile time:

| Define       | Value / Source                          |
|--------------|-----------------------------------------|
| `MAX_RESULT` | `algo::kheavyhash::MAX_RESULT`          |
| `__AMDGCN__` | compiler-provided; gates `v_dot4` path  |

All six variants share the same 6-arg signature
`(matrix, header, target, timestamp, startNonce, result)`, so only the kernel
name and the LDS/matmul/keccak internals differ between runs.

Enable it in `sources/benchmark/config.json` under `amd.algorithms.kheavyhash`
(`enabled: true`), then run `benchmark.exe config.json`. The benchmark is
**synchronous** (one `enqueue` + `finish` per launch, no double-buffering), so the
first launches measure the GPU ramping from idle clocks — let it run the full loop
for a steady-state reading.

---

## Kernel Variants

### LM0 — Reference (correctness baseline)

`kHeavyHash_lm0`. One work-item per nonce, a full keccak-f[1600] per hash, the
matrix re-read from global memory, no LDS caching or vectorisation. Intentionally
**not optimised** — it is the bit-exact reference every other variant is gated
against, and the baseline the gains are measured from.

### LM1 — LDS matrix staging

`kHeavyHash_lm1`. Stage the 64×64 matrix in LDS (`__local uchar[4096]`) once per
workgroup, killing the per-nonce global re-read. Drops VGPR 129 → 94 and raises
occupancy from ~11 to the full 16 waves/SIMD — but only buys **+5%**, because the
baseline was *not* occupancy-bound. It is compute-bound, so extra waves barely
help.

### LM2 — `v_dot4_u32_u8` matmul

`kHeavyHash_lm2`. The 64×64 nibble matmul becomes integer dot products.
gfx1201's OpenCL compiler exposes no `cl_khr_integer_dot_product`, but
`__builtin_amdgcn_udot4` lowers to `v_dot4_u32_u8` (guarded by `__AMDGCN__`, with a
bit-identical scalar fallback for POCL/CI). 4096 nibble-MACs/nonce → 1024 hardware
dot ops. This is the **first real win, +22%** over LM1.

### LM3 — register-resident keccak (tied)

`kHeavyHash_lm3`. Unrolled rho/pi, register-resident keccak on top of LM2. **Ties
LM2** — after the matmul the kernel is keccak-*work* bound, not keccak-*unrolling*
bound, so re-rolling the permutation into registers buys nothing. Retained in-tree
as a recorded negative result.

### LM4 — powHash keccak midstate (shipped)

`kHeavyHash_lm4`. In powHash only `state[9]` (the nonce) varies per nonce;
`prePowHash` + `timestamp` are fixed for the job. So the first keccak's round-1
(state init, the 32-byte absorb, and the theta) is computed **once per workgroup**
(thread 0 → LDS), and each nonce just folds its nonce into 11 lanes and runs from
round-1 ρ/π. This is the **biggest single win, +19% over LM2** (→ **1.46×**
overall). The heavy (second) keccak depends on the matmul output and cannot be
hoisted, so it stays full — which is why the ceiling is ~1.46×, not 2×.
`ResolverAmdKHeavyHash::buildSearch` selects this kernel.

### LM5 — heavy keccak round-0 hoist (tied)

`kHeavyHash_lm5`. The heavy keccak absorbs only 32 bytes into the constant
`HEAVY_INITIAL_STATE`, so its round-0 theta column-parity is partly constant —
trimming round-0 parity reduction from 20 XORs to 4. **Ties LM4 within noise**
(+0.31% mean over five back-to-back idle-GPU runs, below the ~0.7% run-to-run
spread). Why the LM4 trick does not transfer: the nonce touches a *single* column
of powHash (so the whole round-0 theta hoisted), but the matmul product fills lanes
0..3 — one element in *four* of the five columns. Every round-0 theta D-value
depends on the input, nothing hoists past it, and chi's nonlinearity blocks pushing
the input further. This puts the heavy keccak at its floor and **confirms the
~1.45× ceiling**. Retained in-tree as the recorded second-keccak experiment.

---

## Optimisation Progression Summary

RX 9070 XT (RDNA4 / gfx1201), grid 8192 × 256, 200 loops, steady-state median
(first 40 ramp iterations dropped). All six kernels are gated **bit-identical** by
the OpenCL KAT (`opencl_kat_test.cpp`, run on the GPU via `unit_test.exe`).

| Kernel          | Hashrate (steady median) | vs baseline | Lever                                                        |
|-----------------|--------------------------|-------------|--------------------------------------------------------------|
| `kHeavyHash_lm0`| 393 MH/s                 | 1.00×       | reference (one work-item/nonce, matrix re-read from global)  |
| `kHeavyHash_lm1`| 414 MH/s                 | 1.05×       | matrix staged in LDS (`__local uchar[4096]`)                 |
| `kHeavyHash_lm2`| 482 MH/s                 | 1.23×       | + `v_dot4_u32_u8` matmul (`__builtin_amdgcn_udot4`)          |
| `kHeavyHash_lm3`| 481 MH/s                 | 1.23×       | + register-resident (unrolled rho/pi) keccak — no gain       |
| `kHeavyHash_lm4`| **573 MH/s**             | **1.46×**   | + powHash keccak **midstate** (per-job round-1 hoisted)      |
| `kHeavyHash_lm5`| 574 MH/s                 | 1.46×       | + heavy keccak round-0 constant-parity hoist (ties lm4)      |

The optimisation pass (lm0 → lm4) took the kernel **393 → 573 MH/s (+46%)**. Two
levers carried it: the `v_dot4` matmul (lm2, +22%) and the powHash keccak midstate
(lm4, +19% on top). LDS staging (lm1) and keccak unrolling (lm3) were each
marginal.

### RGA static analysis (gfx1201, offline)

| Kernel          | VGPR | → waves/SIMD       | global_load | `v_dot4` | LDS B | spills |
|-----------------|------|--------------------|-------------|----------|-------|--------|
| `kHeavyHash_lm0`| 129  | ~11 (VGPR-bound)   | 22          | 0        | 8192  | 0      |
| `kHeavyHash_lm1`| 94   | 16 (max)           | 7           | 0        | 12288 | 0      |
| `kHeavyHash_lm2`| 84   | 16                 | 7           | 32       | 12288 | 0      |
| `kHeavyHash_lm3`| 83   | 16                 | 7           | 32       | 12288 | 0      |
| `kHeavyHash_lm4`| 90   | 16                 | 7           | 32       | 12488 | 0      |
| `kHeavyHash_lm5`| 90   | 16                 | 7           | 32       | 12488 | 0      |

---

## AMD-Specific Considerations

- **ALU-bound, not occupancy-bound.** LDS staging alone (lm1) raised occupancy from
  ~11 to the full 16 waves/SIMD for only +5%. The bottleneck is integer ALU
  throughput (matmul + two keccak permutations), so the wins came from doing *less*
  ALU work (dot-product matmul, keccak midstate), not from more parallelism.
- **`v_dot4_u32_u8`.** The single largest hardware lever. gfx1201 exposes no
  `cl_khr_integer_dot_product`, so the kernel calls `__builtin_amdgcn_udot4`
  directly under `__AMDGCN__`, with a scalar fallback that keeps POCL/CI
  bit-identical.
- **Keccak midstate.** Hoisting the per-job-constant part of the *first* keccak
  (lm4) is the dominant software lever. The *second* (heavy) keccak is fully
  nonce-dependent and irreducible (lm5), which fixes the ceiling at ~1.45×: two
  essentially-irreducible 24-round permutations plus the matmul, all at max
  occupancy (16 waves/SIMD) with no spills.
- **Benchmark hygiene.** These numbers require an otherwise-idle GPU. A concurrent
  workload on the same card halves and de-orders the results; a load that resumes
  mid-run corrupts the later kernels. Sanity-check that `kHeavyHash_lm0` lands near
  ~393 MH/s and `lm2 > lm1` before trusting a run.
