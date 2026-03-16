# KAWPOW — NVIDIA CUDA Optimisation Work

## Overview

This document tracks the successive CUDA kernel optimisations developed for KAWPOW on NVIDIA GPUs. Each variant (`lm1` through `lm11`) introduces one or more targeted changes. The benchmark runs each kernel 10 times with 256 threads × 1024 blocks.

Benchmark entry point: [sources/benchmark/nvidia/kawpow.cpp](../../sources/benchmark/nvidia/kawpow.cpp)
Kernel sources: [sources/benchmark/cuda/kawpow/](../../sources/benchmark/cuda/kawpow/)

---

## Reference Implementations

Before the custom variants, two reference implementations from the open-source kawpowminer project are benchmarked as a baseline:

| Label              | Description                              |
|--------------------|------------------------------------------|
| `kawpowminer_1`    | First kawpowminer reference kernel       |
| `kawpowminer_2`    | Second kawpowminer reference kernel      |

---

## Kernel Variants

### LM1 — Baseline (No Shared Memory)

**Optimisation**: None. This is the starting point.

- One nonce per thread.
- All 16 lanes processed sequentially inside each thread.
- All DAG accesses go directly to global memory.
- No shared memory used for header caching.

This kernel establishes the performance floor and correctness reference.

---

### LM2 — Shared Memory Header DAG Cache

**Optimisation**: Cache the first `MODULE_CACHE` (4096) DAG items in shared memory.

```cuda
__shared__ uint32_t header_dag[MODULE_CACHE];
initialize_header_dag(header_dag, dag);
```

- All threads in a block cooperate to fill `header_dag` from global memory once at startup.
- `loop_math()` receives `header_dag` and reads cached values instead of going to global memory for the first 4096 items.
- Reduces global memory bandwidth for the most frequently accessed DAG region.

---

### LM3 — Compiler Unroll Hints

**Optimisation**: Add `#pragma unroll` directives to hot loops.

- `sha3()`, `fill_hash()`, `reduce_hash()`, and `initialize_header_dag()` loops annotated with explicit unroll hints.
- Helps the compiler unroll fixed-iteration loops and reduce branch overhead.
- Retains the shared memory header DAG cache from LM2.

---

### LM4 — Register Pressure Investigation

**Optimisation target**: Reduce register usage to 64 registers per thread (improves occupancy).

- Functionally identical to LM3 in the benchmark.
- Represents an attempt to guide the compiler toward lower register usage via code structure.
- The shared memory cache and unroll hints from LM2/LM3 are retained.

---

### LM5 — `__threadfence_block` on DAG Load

**Optimisation**: Add a memory fence after each DAG lookup in `loop_math`.

```cuda
uint4 entries = dag[dagIndex];
__threadfence_block();
sequence_math_random(header_dag, hash, &entries);
```

- `__threadfence_block()` ensures the DAG load is visible to all threads in the block before the math sequence begins.
- Tests whether explicit memory ordering improves instruction scheduling and memory pipeline utilisation.

---

### LM6 — `__threadfence_block` on Header Initialisation

**Optimisation**: Extend the memory fence to the shared memory initialisation phase.

```cuda
// Inside initialize_header_dag()
header_dag[indexDAG] = itemDag;
__threadfence_block();
```

- Both the header DAG fill and each DAG lookup in `loop_math` are now fenced.
- Tests more aggressive memory ordering to determine its impact on throughput.

---

### LM7 — Multi-Nonce Loop (10× per Thread)

**Optimisation**: Each thread processes 10 nonces in a single kernel launch.

```cuda
for (uint32_t i = 0u; i < 10u; ++i) {
    // full hash for current nonce
    nonce += nonceComputed;  // nonceComputed = blocks × threads
}
```

- Amortises kernel launch overhead across 10 nonce evaluations per thread.
- The benchmark sets a `multiplicator` of 10 to account for the increased throughput.
- Uses `if (bytes == 0ull)` for result validation in this variant.

---

### LM8 — Multi-Nonce Loop with Corrected Validation

**Optimisation**: Same multi-nonce loop as LM7, with standard validation logic restored.

- 10-iteration nonce loop identical to LM7.
- Validation uses `if (bytes < 1ull)` (standard difficulty comparison) instead of the equality check in LM7.
- Includes shared memory header DAG cache and `__threadfence_block` in `loop_math`.
- Combines the throughput gain of LM7 with correct behaviour.

---

### LM9 — Read-Only Cache (`__ldg`) Instead of Shared Memory

**Optimisation**: Remove shared memory caching; use the hardware L2/texture read-only cache via `__ldg`.

```cuda
// No shared memory allocation
uint32_t const* const header_dag = (uint32_t*)dag;

// In sequence_math_random_cache_only():
mix[17] = ror_u32(mix[17], 24) ^ __ldg(&header_dag[dag_offset]);
```

- Eliminates the `initialize_header_dag()` overhead entirely.
- `__ldg()` routes accesses through the read-only data cache, which is persistent across warps.
- Tests whether hardware-managed caching outperforms the explicit shared memory strategy.

---

### LM10 — Warp-Level Parallelism with Shuffle

**Optimisation**: Exploit warp-level intrinsics to parallelise lane computation across threads.

```cuda
// Thread assignment
uint32_t const lane_id = threadIdx.x & LANE_ID_MAX;   // 0-15
uint32_t const warp_id = threadIdx.x / LANES;

// Broadcast seed across warp lanes
uint32_t lane_lsb = __shfl_sync(0xffffffff, lsb, target_lane, LANES);
uint32_t lane_msb = __shfl_sync(0xffffffff, msb, target_lane, LANES);
```

- Instead of one thread iterating over all 16 lanes sequentially, 16 threads (one warp) each handle one lane concurrently.
- `__shfl_sync` broadcasts seed values within the warp without shared memory.
- A `shared_digest` buffer is used to transpose digest results across the warp.
- This version requires dividing the block count by 16 (`setDivisor(16u)`) since each nonce now consumes 16 threads.

---

### LM11 — Explicit Block Scoping for Compiler Guidance

**Optimisation**: Wrap major computation phases in explicit `{ }` blocks.

- Header initialisation, seed creation, and lane processing are each enclosed in their own scope.
- Helps the compiler narrow variable lifetimes and improve register allocation.
- Retains shared memory header DAG cache and `__threadfence_block` from LM6.
- Marked as a TODO to further reduce register count.

---

## Optimisation Progression Summary

| Kernel | Key Change                                              |
|--------|---------------------------------------------------------|
| LM1    | Baseline — no shared memory, sequential lanes           |
| LM2    | Shared memory header DAG cache                          |
| LM3    | Compiler unroll hints on hot loops                      |
| LM4    | Register pressure investigation (structure only)        |
| LM5    | `__threadfence_block` after DAG load in `loop_math`     |
| LM6    | `__threadfence_block` also in `initialize_header_dag`   |
| LM7    | 10-nonce loop per thread (reduced kernel launch overhead)|
| LM8    | 10-nonce loop + corrected validation                    |
| LM9    | `__ldg` read-only cache, removes shared memory entirely |
| LM10   | Warp-level parallelism via `__shfl_sync`                |
| LM11   | Explicit scoping for register/compiler optimisation     |
