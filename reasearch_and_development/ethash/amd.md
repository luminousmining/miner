# ETHASH — AMD OpenCL Optimisation Work

## Overview

This document covers the OpenCL search-kernel optimisation developed for Ethash on
AMD GPUs (RDNA/RDNA4). Two variants are benchmarked head-to-head:

- **`ethash_search_baseline`** — the original kernel, using a full work-group
  `barrier(CLK_LOCAL_MEM_FENCE)` at every lane-exchange point.
- **`ethash_search_subgroup`** — identical, except the three intra-wavefront
  lane exchanges use `sub_group_barrier(CLK_LOCAL_MEM_FENCE)` instead. Each
  thread only ever reads its own `LANE_PARALLEL`-lane group's slots, so a
  work-group-wide barrier is stronger than required; a sub-group barrier
  synchronises just the wavefront and lets independent wavefronts proceed.

Both variants are byte-for-byte copies of the production `ethash_search.cl`
(baseline = pre-change, subgroup = post-change), so the benchmark measures
exactly what ships. The only delta between the two files is `barrier` →
`sub_group_barrier` on the three lane-exchange fences.

Benchmark entry point: [sources/benchmark/amd/ethash.cpp](../../sources/benchmark/amd/ethash.cpp)
Kernel sources: [sources/benchmark/opencl/ethash/](../../sources/benchmark/opencl/ethash/)

---

## Benchmark Setup

Each variant is compiled at runtime via `KernelGeneratorOpenCL`, which injects
defines before building the OpenCL source. The kernel function is always
`ethash_search`; only the source file differs between variants.

- **Group size**: 256 work-items (`GROUP_SIZE`)
- **Worker group count**: 8192 (`blocks`)
- **Lane parallelism**: `LANE_PARALLEL` = 8

Key defines injected at compile time (mirroring `ResolverAmdEthash::buildSearch`):

| Define             | Value / Source                              |
|--------------------|---------------------------------------------|
| `GROUP_SIZE`       | 256                                         |
| `DAG_NUMBER_ITEM`  | `dagContext.dagCache.numberItem`            |
| `LANE_PARALLEL`    | 8                                           |
| `LEN_SEED`         | 4                                           |
| `LEN_STATE`        | 25                                          |
| `LEN_HASHES`       | `(GROUP_SIZE / LANE_PARALLEL) * LEN_SEED`   |
| `LEN_WORD0`        | `GROUP_SIZE`                                |
| `LEN_REDUCE`       | `GROUP_SIZE`                                |
| `LEN_SWAPPER`      | `GROUP_SIZE / LANE_PARALLEL`                |
| `LEN_KECCAK`       | 24                                          |
| `MAX_KECCAK_ROUND` | 23                                          |

The DAG is built once before the search variants run, via the shared
`ethash_build_dag` kernel (`kernel/ethash/ethash_dag.cl`).

Run with:

```sh
# enable amd + ethash in sources/benchmark/config.json, then:
./bin/benchmark
```

---

## Methodology

Per `BENCHMARK.md`: each variant runs `loop` iterations (default 10); the
dashboard reports the mean kernel hashrate. Because both variants share an
identical harness, DAG, grid, and accounting, the **relative** delta is a valid
measure of the optimisation even though the absolute figure is a per-launch
kernel rate rather than an end-to-end miner hashrate.

For an end-to-end cross-check, an interleaved A/B of full-miner runs
(alternating baseline/optimised builds, N rounds each, reading reported
hashrate over a fixed window) cancels thermal/clock drift.

---

## Results

> TODO: populate from `./bin/benchmark` on the target GPU. Numbers below are
> placeholders.

**Device:** RX 9070 XT (RDNA4, gfx1201) — _fill in driver / ROCm version_

| variant                   | mean (MH/s) | Δ vs baseline |
|---------------------------|-------------|---------------|
| `ethash_search_baseline`  | _TODO_      | —             |
| `ethash_search_subgroup`  | _TODO_      | _TODO_        |

End-to-end interleaved A/B (full miner, `ethw.2miners.com`):

| build      | per-round (MH/s)        | mean   |
|------------|-------------------------|--------|
| baseline   | _TODO_                  | _TODO_ |
| optimised  | _TODO_                  | _TODO_ |
