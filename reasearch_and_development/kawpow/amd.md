# KAWPOW — AMD OpenCL Optimisation Work

## Overview

This document covers the OpenCL kernel optimisations developed for KAWPOW on AMD GPUs. Four variants (`lm1` through `lm4`) have been implemented, each exploring different memory strategies suited to AMD's GCN/RDNA architecture (wavefront size, LDS usage, sub-group operations).

Benchmark entry point: [sources/benchmark/amd/kawpow.cpp](../../sources/benchmark/amd/kawpow.cpp)
Kernel sources: [bin/kernel/kawpow/](../../bin/kernel/kawpow/)

---

## Benchmark Setup

Each kernel is compiled at runtime via `KernelGeneratorOpenCL`, which injects defines before building the OpenCL source. The kernel is invoked with:

- **Group size**: 256 work-items
- **Worker group count**: 1024
- **Work-item collaboration**: `algo::progpow::LANES` (16)

Key defines injected at compile time:

| Define                  | Value / Source                                         |
|-------------------------|--------------------------------------------------------|
| `GROUP_SIZE`            | 256                                                    |
| `REGS`                  | `algo::progpow::REGS` (32)                             |
| `WAVEFRONT`             | Device wavefront width (AMD: typically 64)             |
| `WORK_ITEM_COLLABORATE` | `LANES` (16) — work-items sharing one nonce            |
| `COUNT_DAG`             | `algo::progpow::COUNT_DAG` (64)                        |
| `MODULE_CACHE`          | `algo::progpow::MODULE_CACHE` (4096)                   |
| `DAG_SIZE`              | `dagContext.dagCache.numberItem / 2`                   |
| `BATCH_GROUP_LANE`      | `GROUP_SIZE / WORK_ITEM_COLLABORATE` (16)              |
| `SHARE_SEED_SIZE`       | `BATCH_GROUP_LANE` (16)                                |
| `SHARE_HASH0_SIZE`      | `BATCH_GROUP_LANE` (16)                                |
| `SHARE_FNV1A_SIZE`      | `GROUP_SIZE` (256)                                     |

The DAG must be built before any benchmark kernel runs. The DAG build kernel (`ethash_build_dag`) is compiled separately and executed once:

```cpp
generator.addDefine("DAG_LOOP", algo::kawpow::DAG_ITEM_PARENTS / 4u / 4u);
generator.appendFile("kernel/ethash/ethash_dag.cl");
```

---

## Work-Item Collaboration Model

Unlike the CUDA variants where one thread handles all 16 lanes sequentially, the OpenCL variants assign **16 work-items to cooperate on a single nonce**. Each work-item handles one lane:

```
Work-items 0-15  → nonce 0
Work-items 16-31 → nonce 1
...
```

This is controlled by `WORK_ITEM_COLLABORATE = LANES = 16`. Within each collaboration group, work-items share data through **LDS (Local Data Share)** buffers.

---

## Kernel Variants

### LM1 — Full LDS Caching (Maximum Shared Memory)

**Strategy**: Cache as much data as possible in LDS.

LDS buffers allocated:

```opencl
__local uint  header_dag[MODULE_CACHE];          // 4096 items from DAG
__local ulong share_msb_lsb[SHARE_SEED_SIZE];    // seed (lsb/msb) per lane group
__local uint  share_hash0[SHARE_HASH0_SIZE];     // hash[0] per lane group
__local uint  share_fnv1a[SHARE_FNV1A_SIZE];     // FNV1a intermediate per work-item
```

- `initialize_header()` distributes the DAG header load across all work-items in the group.
- Seed values and hash[0] are exchanged through LDS to avoid redundant computation.
- Uses `sequence_dynamic_local()` — the math sequence reads from `header_dag` in LDS.

This is the highest-occupancy approach for AMD hardware where LDS bandwidth is fast.

---

### LM2 — Direct DAG Access with Sub-Group Operations

**Strategy**: Remove LDS header caching; use sub-group (wavefront) operations for seed exchange.

```opencl
// No header_dag LDS buffer
uint4 const entries = ((uint4*)dag)[dag_index];
```

- Seed values (`lsb`/`msb`) are exchanged using `get_sub_group_local_id()` and sub-group broadcast — no LDS required for this.
- Uses `sequence_dynamic()` — the math sequence accesses the DAG directly through L2 cache.
- Trades LDS usage for reliance on hardware cache and sub-group communication.

Sub-group operations map directly to AMD wavefront shuffle instructions, avoiding LDS synchronisation overhead for the seed exchange.

---

### LM3 — Header Cache + Sub-Group Seed Exchange

**Strategy**: Restore the LDS header DAG cache; keep sub-group operations for seed exchange.

```opencl
__local uint header_dag[MODULE_CACHE];
initialize_header(header_dag, dag);
```

- LDS header caching is re-enabled from LM1.
- Sub-group operations (from LM2) are used for the seed broadcast.
- `sequence_dynamic_local()` reads from LDS header cache.
- This combines the bandwidth benefit of caching with the low-overhead sub-group communication.

---

### LM4 — LDS Header Cache (Refinement of LM3)

**Strategy**: Incremental refinement of LM3.

- Same LDS header cache and `initialize_header()` as LM3.
- Same `sequence_dynamic_local()` access pattern.
- Represents a tuning iteration on top of LM3 — code structure or kernel argument layout may differ.

---

## Optimisation Progression Summary

| Kernel | LDS Header Cache | Seed Exchange Method      | DAG Access              |
|--------|------------------|---------------------------|-------------------------|
| LM1    | Yes              | LDS (`share_msb_lsb`)     | `sequence_dynamic_local` (LDS) |
| LM2    | No               | Sub-group broadcast       | `sequence_dynamic` (L2 cache)  |
| LM3    | Yes              | Sub-group broadcast       | `sequence_dynamic_local` (LDS) |
| LM4    | Yes              | Sub-group broadcast       | `sequence_dynamic_local` (LDS) |

---

## AMD-Specific Considerations

- **Wavefront width**: AMD GPUs execute 64 work-items in lockstep (vs. 32 on NVIDIA). The `WAVEFRONT` define is read from the device at runtime and injected into the kernel to allow wavefront-aware scheduling.
- **LDS vs L2**: AMD LDS bandwidth is very high within a workgroup. Caching the 4096-item DAG header in LDS (LM1/LM3/LM4) exploits this effectively for the header region.
- **Sub-group operations**: `get_sub_group_local_id()` and related built-ins map to AMD DS_SWIZZLE or `v_readlane` instructions — faster than writing to and reading from LDS for simple broadcasts.
- **DAG build**: KAWPOW uses `DAG_ITEM_PARENTS = 512` (vs. 256 for Ethash). The DAG build loop is adjusted accordingly: `DAG_LOOP = DAG_ITEM_PARENTS / 4 / 4`.

---

## Production kernel A/B — LDS coalesce + `sub_group_barrier`

Distinct from the `lm1`–`lm4` exploration above, this benchmarks the **production**
`progpow_search` kernel (the one the miner actually runs) before vs after the
optimisation in this PR. Both variants are byte-for-byte copies of the production
`progpow.cl`; the rest of the kernel (result struct, `kawpow_functions.cl`, and the
per-period generated math sequence) is shared, so the delta is exactly the change:

- **`progpow_lm_0`** — original: strided AoS LDS header store (4 words/lane,
  bank-conflicting) + full work-group `barrier(CLK_LOCAL_MEM_FENCE)` at the three
  lane-exchange points.
- **`progpow_lm_1`** — this PR: coalesced one-uint-per-lane LDS store
  (bank-conflict-free, coalesced global read) + `sub_group_barrier` on the three
  intra-wavefront exchanges.

Benchmark entry point: [sources/benchmark/amd/progpow.cpp](../../sources/benchmark/amd/progpow.cpp)
(assembly mirrors `ResolverAmdProgPOW::buildSearch`; config algo key `progpow`).
Kernel sources: [sources/benchmark/opencl/progpow/](../../sources/benchmark/opencl/progpow/)

### Results

**Device:** RX 9070 XT (RDNA4, gfx1201), Windows host, cross-compiled
(`windows-amd-cross`). 5 runs × 10 iterations per variant, `threads=256`,
`blocks=1024`. Per-run median, paired within run (see the Ethash doc's
Methodology — absolute rate drifts ±5–7% run-to-run, so only the within-run
baseline↔subgroup ratio is meaningful).

| variant             | median-of-medians (MH/s) | per-run median range |
|---------------------|--------------------------|----------------------|
| `progpow_lm_0`  | 37.91                    | 34.46 – 38.16        |
| `progpow_lm_1`  | 38.60                    | 37.02 – 38.93        |

Per-run paired Δ (subgroup vs baseline, same run): **+7.4%, +1.9%, +1.1%, +2.5%,
−1.3%** → **+2.3% mean / +1.8% median**, positive in 4/5 runs. The coalesced LDS
store + `sub_group_barrier` is consistently a small win; the outliers (+7.4% from
a cold-cache baseline, −1.3% from a low subgroup sample) are the run-to-run noise
the pairing is designed to bound. Notably `progpow_lm_0` runs *first* (cooler)
each run, so the gain shows up despite a slight thermal handicap.
