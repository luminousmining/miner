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
