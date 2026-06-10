# Benchmark — Architecture & Workflow

## Overview

The benchmark is a standalone executable that measures the performance of GPU mining kernels across multiple algorithms and hardware vendors. Its goal is to compare kernel variants side by side and guide optimisation decisions.

Entry point: [sources/benchmark.cpp](sources/benchmark.cpp)

```cpp
int main() {
    benchmark::BenchmarkWorkflow bench{ true, true }; // NVIDIA + AMD
    if (false == bench.initializeDevices()) { return 1; }
    bench.run();
    return 0;
}
```

---

## Directory Structure

```
sources/
├── benchmark.cpp                   # Main entry point
└── benchmark/
    ├── workflow.hpp / workflow.cpp  # Orchestration class
    ├── result.hpp                   # Result data structures
    ├── nvidia.hpp / nvidia.cpp      # NVIDIA device initialisation
    ├── amd.hpp / amd.cpp            # AMD device initialisation
    ├── nvidia/                      # Per-algorithm NVIDIA benchmarks
    │   ├── kawpow.cpp               # 13 kernel variants
    │   ├── ethash.cpp               # 4 kernel variants
    │   ├── keccak.cpp               # 9 kernel variants
    │   ├── fnv1.cpp                 # 2 kernel variants
    │   ├── autolykos.cpp            # 3 kernel variants
    │   └── ethash_light_cache.cpp   # 3 kernel variants
    ├── amd/
    │   └── kawpow.cpp               # 4 OpenCL kernel variants
    └── cuda/
        ├── kernels.hpp              # Central CUDA kernel declarations
        ├── kawpow/                  # KAWPOW CUDA kernel sources
        ├── ethash/                  # Ethash CUDA kernel sources
        ├── keccak/                  # Keccak-f800 CUDA kernel sources
        ├── fnv1/                    # FNV1 CUDA kernel sources
        └── autolykos_v2/            # Autolykos V2 CUDA kernel sources

bin/kernel/                          # OpenCL kernel sources (.cl)
├── kawpow/
├── ethash/
└── common/
```

---

## BenchmarkWorkflow Class

[sources/benchmark/workflow.hpp](sources/benchmark/workflow.hpp)

The `BenchmarkWorkflow` class is the central orchestrator. It owns device state, timing, and result collection.

### Key Members

| Member | Type | Role |
|---|---|---|
| `propertiesNvidia` | `PropertiesNvidia` | CUDA device, context, stream |
| `propertiesAmd` | `PropertiesAmd` | OpenCL device, context, queue |
| `stats` | `statistical::Statistical` | Timing and hashrate computation |
| `dashboards` | `vector<Dashboard>` | One table per algorithm family |
| `snapshots` | `vector<Snapshot>` | Raw results for JSON report |
| `blocks` / `threads` | `uint32_t` | Current kernel grid configuration |
| `nonceComputed` | `uint32_t` | `blocks × threads` — nonces per launch |

### Execution Flow

```
initializeDevices()
    ├── CUDA: device selection → context → stream
    └── OpenCL: platform → device → context → queue

run()
    ├── runNvidia()
    │   ├── runNvidiaKeccak()
    │   ├── runNvidiaFnv1()
    │   ├── runNvidiaEthashLightCache()
    │   ├── runNvidiaEthash()
    │   ├── runNvidiaAutolykosv2()
    │   └── runNvidiaKawpow()
    ├── runAmd()
    │   └── runAmdKawpow()
    ├── display all dashboards
    └── writeReport() → benchmark.json
```

---

## Running a Benchmark — Step by Step

Each algorithm benchmark follows the same pattern:

### 1. Create a Dashboard

```cpp
common::Dashboard dashboard{ createNewDashboard("[NVIDIA] KAWPOW") };
```

### 2. Allocate Device Memory

```cpp
algo::hash1024* dagHash{ nullptr };
CU_ALLOC(&dagHash, dagItems * algo::LEN_HASH_1024);
```

### 3. Copy Test Data to Device

```cpp
CUDA_ER(cudaMemcpy(headerHash->bytes, header.bytes, algo::LEN_HASH_256, cudaMemcpyHostToDevice));
```

### 4. Run the Benchmark with `RUN_BENCH`

```cpp
RUN_BENCH(
    "kawpow: lm2",
    10,      // loop count
    256,     // threads per block
    1024,    // blocks
    kawpow_lm2(stream, result, headerHash->word32, dagHash->word32, blocks, threads)
)
```

`RUN_BENCH` expands to:
- Call `setGrid(threads, blocks)` to configure the grid.
- For each of the `loop` iterations:
  - `startChrono(name)` — reset stats, record start time.
  - Execute the kernel function.
  - `stopChrono(dashboard)` — compute elapsed time, calculate hashrate, add a row to the dashboard.
- Reset the multiplicator/divisor to 1 after the loop.

### 5. Reset the Result Buffer

```cpp
BENCH_INIT_RESET_RESULT(result);
```

Clears the `t_result` structure between kernel variants so a previous valid nonce does not pollute the next run.

### 6. Free Device Memory

```cpp
CU_SAFE_DELETE(dagHash);
CU_SAFE_DELETE(headerHash);
CU_SAFE_DELETE_HOST(result);
```

### 7. Register the Dashboard

```cpp
dashboards.emplace_back(dashboard);
```

---

## Timing and Hashrate

### Chrono

`startChrono` / `stopChrono` wrap each kernel launch with `std::chrono::system_clock` at microsecond precision.

### Hashrate Calculation

```
hashrate = (nonceComputed × multiplicator / divisor × kernelCount) / elapsed_μs × 1e6
```

- `nonceComputed = blocks × threads` — nonces evaluated per launch.
- `multiplicator` — increased when one kernel launch covers multiple nonces (e.g. ×10 for lm7/lm8).
- `divisor` — decreased when one nonce requires multiple threads (e.g. ÷16 for lm10 warp parallelism).

### Hashrate Display

Hashrates are auto-scaled and printed as `H`, `KH`, `MH`, `GH`, `TH`, etc. at 2 decimal places.

---

## Result Structures

[sources/benchmark/result.hpp](sources/benchmark/result.hpp)

| Structure | Fields | Used For |
|---|---|---|
| `t_result` | `found`, `count`, `nonce` (64-bit) | Standard nonce result |
| `t_result_32` | `found`, `count`, `mix[8]` (32-bit) | 256-bit hash results |
| `t_result_64` | `found`, `count`, `mix[8]` (64-bit) | 512-bit hash results |

Results are stored in CUDA pinned host memory (`CU_ALLOC_HOST`) for zero-copy access from both CPU and GPU.

---

## Macros Reference

### Benchmark Macros (`workflow.hpp`)

| Macro | Purpose |
|---|---|
| `RUN_BENCH(name, loop, threads, blocks, fn)` | Time and record `fn` over `loop` iterations |
| `BENCH_INIT_RESET_RESULT(result)` | Allocate / zero a `t_result` |
| `BENCH_INIT_RESET_RESULT_32(result)` | Allocate / zero a `t_result_32` |
| `BENCH_INIT_RESET_RESULT_64(result)` | Allocate / zero a `t_result_64` |

### Memory Macros (`common/custom.hpp`)

| Macro | Purpose |
|---|---|
| `CU_ALLOC(ptr, size)` | `cudaMalloc` with tracking |
| `CU_CALLOC(ptr, size)` | `cudaMalloc` + `cudaMemset` |
| `CU_ALLOC_HOST(ptr, size)` | Pinned host allocation |
| `CU_SAFE_DELETE(ptr)` | `cudaFree` + null |
| `CU_SAFE_DELETE_HOST(ptr)` | Pinned host free + null |
| `IS_NULL(ptr)` | Log error and return false if null |
| `CUDA_ER(call)` | Assert CUDA call returned no error |

### Rate Adjustment

| Method | Usage |
|---|---|
| `setMultiplicator(n)` | Kernel processes `n` nonces per thread (e.g. ×10 in lm7/lm8) |
| `setDivisor(n)` | `n` threads share one nonce (e.g. ÷16 in lm10) |

---

## AMD OpenCL Benchmarks

AMD kernels are compiled at runtime via `KernelGeneratorOpenCL`. This class:
1. Collects `#define` values and `#include` directives.
2. Appends the main `.cl` kernel file.
3. Calls `clBuildProgram` against the selected device.
4. Exposes the built `cl::Kernel` for argument binding and dispatch.

This allows defines such as `GROUP_SIZE`, `WAVEFRONT`, `DAG_SIZE`, and `WORK_ITEM_COLLABORATE` to be set at runtime based on the actual device properties and DAG context.

The AMD workflow requires an additional DAG build step before any mining kernel can run:

```
Build DAG kernel (ethash_build_dag.cl)
    → enqueueNDRangeKernel (fills dagCache from lightCache)
    → clFinish()

For each kawpow_lmN kernel:
    → KernelGeneratorOpenCL::build()
    → setArg(dag, result, header, startNonce)
    → enqueueNDRangeKernel loop
    → clFinish()
    → stopChrono(dashboard)
```

---

## Algorithms Covered

### NVIDIA (CUDA)

| Algorithm | Variants | Notes |
|---|---|---|
| Keccak-f800 | 9 (lm1–lm9) | Core hash function used by Ethash / KAWPOW |
| FNV1 | 2 (lm1–lm2) | Mix hash; lm2 uses `__umulhi` |
| Ethash light cache | 3 (lm1–lm3) | DAG generation from light cache |
| Ethash | 4 (base, ethminer, lm1, lm2) | Full Ethash mining kernel |
| Autolykos V2 | 3 (mhssamadi, lm1, lm2) | Ergo mining algorithm |
| KAWPOW | 13 (kawpowminer ×2, lm1–lm11) | Ravencoin mining algorithm |

### AMD (OpenCL)

| Algorithm | Variants | Notes |
|---|---|---|
| KAWPOW | 4 (lm1–lm4) | LDS caching and sub-group strategies |

---

## Output

### Console

One formatted table per algorithm family, printed after all runs complete.

### JSON Report

Written to `benchmark.json` in the working directory, structured as:

```json
{
  "nvidia": [
    { "name": "kawpow: lm1", "threads": 256, "blocks": 1024, "perform": 27140000.0 },
    ...
  ],
  "amd": [
    { "name": "kawpow_lm1", "threads": 256, "blocks": 1024, "perform": 0.0 },
    ...
  ]
}
```

---

## Reference Results — NVIDIA GeForce RTX 4070 Ti

Measured on 2026-03-16, 10 iterations per kernel, config: `sources/benchmark/config.json`.

### Keccak-f800 — 128 threads × 1024 blocks

| Kernel | Hashrate (steady) |
|---|---|
| lm1 | ~5.96 GH |
| lm2 | ~5.96 GH |
| lm3 | ~5.96 GH |
| lm4 | ~5.96 GH |
| lm5 | ~5.96 GH |
| lm6 | ~6.24 GH |
| lm7 | ~5.96 GH |
| lm8 | ~5.96 GH |
| lm9 | ~6.24 GH |

> First iteration is always slower (cold GPU / JIT warm-up). Steady-state values shown above.

### FNV1 — 1024 threads × 8192 blocks

| Kernel | Hashrate (steady) |
|---|---|
| lm1 | ~158.28 GH |
| lm2 | ~158.28 GH |

### Ethash Light Cache — 1 block × 1409017 threads

| Kernel | Hashrate | Time |
|---|---|---|
| lm2 | 55.41 KH | ~25.4 s |
| lm3 | 56.77 KH | ~24.8 s |

### Ethash — 128 threads × 8192 blocks

| Kernel | Hashrate (steady) |
|---|---|
| base    | ~13.85 MH |
| ethminer | ~52.12 MH |
| lm1     | ~51.88 MH |
| lm2     | ~49.67 MH |

### Autolykos V2 — 64 threads × 131072 blocks

| Kernel | Hashrate (steady) |
|---|---|
| mhssamadi | ~70.30 MH |
| lm1       | ~70.58 MH |
| lm2       | ~70.53 MH |

### KAWPOW — 256 threads × 1024 blocks

| Kernel | Hashrate (steady) | Notes |
|---|---|---|
| kawpowminer_1 | ~27.31 MH | Reference |
| kawpowminer_2 | ~27.21 MH | Reference |
| lm1  | ~27.21 MH | |
| lm2  | ~27.30 MH | |
| lm3  | ~21.37 MH | Slower — unroll disabled |
| lm4  | ~26.80 MH | |
| lm5  | ~27.23 MH | |
| lm6  | ~27.30 MH | |
| lm7  | ~27.34 MH | ×10 nonces/thread — time ×10 |
| lm8  | ~27.30 MH | ×10 nonces/thread — time ×10 |
| lm9  | ~27.15 MH | |
| lm10 | ~16.25 MH | 64 blocks only — warp parallelism, needs tuning |
| lm11 | ~27.35 MH | |
