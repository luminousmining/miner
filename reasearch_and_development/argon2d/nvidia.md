# Argon2d — NVIDIA CUDA Optimisation Work

## Overview

This document tracks the successive CUDA kernel optimisations developed for
Argon2d on NVIDIA GPUs. Each variant (`lm1` through `lm7`) introduces one or more
targeted changes. The benchmark runs each kernel 5 times with
128 blocks × 32 threads (LM1–LM6) or 2048 blocks × 1 thread (LM7).

Argon2d parameters: `t=3` (time cost), `m=8` (memory blocks), `p=1` (parallelism).
Each thread operates on a private 8-block working set (8 KiB of `uint64` words).
The H' initialisation is replaced by a fake seeded fill so the benchmark
measures only G compression + fBlaMka + data-dependent memory traversal.

Benchmark entry point: [sources/benchmark/nvidia/argon2d.cpp](../../sources/benchmark/nvidia/argon2d.cpp)
Kernel sources: [sources/benchmark/cuda/argon2d/](../../sources/benchmark/cuda/argon2d/)

---

## Benchmark Results

| Kernel | Hashrate (steady state) | Key technique                                |
|--------|------------------------|----------------------------------------------|
| LM1    | ~4.1 MH/s              | Baseline                                     |
| LM2    | ~4.4 MH/s              | `#pragma unroll`                             |
| LM3    | ~2.8 MH/s              | Template WITH_XOR + fully unrolled fill loop |
| LM4    | ~3.0 MH/s              | Named registers v0–v15 + macro P and G_MIX  |
| LM5    | ~3.9 MH/s              | Eliminate R[128] array                       |
| LM6    | ~2.9 MH/s              | `__ldg()` + IMAD.WIDE cast                  |
| LM7    | ~1.15 MH/s             | Static shared memory, 1 thread per block     |

---

## Kernel Variants

### LM1 — Baseline

**Optimisation**: None. This is the correctness reference and performance floor.

- 1 thread = 1 complete Argon2d solution.
- `G_block_lm1()` is a `__device__` function with two local arrays:
  `R[128]` (XOR of X and Y) and `Q[128]` (working copy for P permutations).
- `G_mix_lm1()` is a `__device__ __forceinline__` function; fBlaMka mixing
  uses `& 0xFFFFFFFFull` to extract the low 32 bits before multiplying.
- `P_lm1(uint64_t* const v)` takes a pointer to a 16-element array.
- The `withXor` branch (pass > 0) is a runtime `bool` parameter — not elided.
- The fill loop iterates over passes, slices and indices at runtime with full
  `computeRefCol_lm1()` logic (all code paths, all casts, runtime modulo).

**Result: ~4.1 MH/s**

---

### LM2 — Compiler Unroll Hints

**Optimisation**: Add `#pragma unroll` on all fixed-bound loops inside
`G_block_lm2` and `fakeInit_lm2`.

- Functionally identical to LM1.
- `#pragma unroll` added before: the 128-word XOR/copy init, both row and
  column permutation loops, the scatter/gather inner loops (`k = 0..7`), and
  the final write loop.
- The compiler can now fully unroll the G_block body and reduce branch and
  loop-counter overhead.

**Result: ~4.4 MH/s — +7% over LM1**

---

### LM3 — Template WITH_XOR + Fully Unrolled Fill Loop

**Optimisation**: Eliminate the `withXor` runtime branch via template
specialisation, and replace the three-level fill loop with 22 explicit macro
calls.

- `G_block_lm3<bool WITH_XOR>`: the `if (withXor)` branch disappears at
  compile time; two instantiations are generated (`false` for pass 0,
  `true` for passes 1–2).
- `computeRefColFixed_lm3()`: `refArea` and `start` are compile-time constants,
  enabling the phi formula to be fully folded.
- `ARGON2D_LM3_STEP(prevIdx, colIdx, refArea, start, withXor)` macro: one
  block-scoped expression that reads `J1`, calls `computeRefColFixed_lm3`, and
  dispatches the template instantiation.
- The kernel body replaces the 3-level loop with 22 explicit STEP calls
  (6 for pass 0, 8 for pass 1, 8 for pass 2).

```cuda
// Pass 0 — withXor = false, start = 0
ARGON2D_LM3_STEP(1u, 2u, 1u, 0u, false)    // slice=1 idx=0
ARGON2D_LM3_STEP(2u, 3u, 2u, 0u, false)    // slice=1 idx=1
// ...
// Pass 1 — withXor = true
ARGON2D_LM3_STEP(7u, 0u, 5u, 2u, true)     // slice=0 idx=0
// ...
```

**Result: ~2.8 MH/s — SLOWER than LM1 (-32%)**

Expanding 22 full G_block calls inline produces enormous register pressure
and instruction-cache thrash. Each G_block needs 256 `uint64` words of local
storage (R + Q = 2 KB) plus 16 temporary registers for each P call. With 22
calls inlined, the compiler cannot keep the working set in registers and must
spill heavily to local (= off-chip global) memory. The elimination of the loop
and branch overhead is overwhelmed by the spill cost.

---

### LM4 — Named Registers + Macro P and G_MIX

**Optimisation**: Replace the `v[16]` array argument of P with 16 named
scalar variables, and inline G_mix as a macro.

- `G_MIX_LM4(a, b, c, d)`: fBlaMka as a `#define` macro — expands inline
  without a function call.
- `P_LM4(v0..v15)`: 16 named arguments; the macro expands to 8 `G_MIX_LM4`
  calls directly on the named scalars.
- Inside `G_block_lm4`, each row and column permutation explicitly loads 16
  named registers from `Q[]`, applies `P_LM4`, then writes them back.
  This eliminates the `uint64_t v[16]` local array and the pointer-indexed
  access that prevents the compiler from keeping the P working set in registers.
- `R[128]` and `Q[128]` arrays are still present.
- Template `WITH_XOR` and `ARGON2D_LM4_STEP` macro retained from LM3.

```cuda
uint64_t v0 { Q[l * 16u + 0u] };
// ...
uint64_t v15{ Q[l * 16u + 15u] };

P_LM4(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)

Q[l * 16u + 0u] = v0;
// ...
```

**Result: ~3.0 MH/s — +7% over LM3, still -27% vs LM1**

Named registers do help the compiler keep the 16 P-round values in registers,
but `R[]` and `Q[]` remain as 1 KB local arrays each. The 22 inlined G_block
calls still cause extreme register and local-memory pressure. Recovering the
per-P array spill does not offset the cost of full fill-loop unrolling.

---

### LM5 — Eliminate R[128] Array

**Optimisation**: Remove the intermediate `R[128]` array; recompute `X^Y`
in the final write instead of storing it.

- `G_block_lm5`: only `Q[128]` exists; `R[]` is gone.
- Initial XOR: `Q[i] = X[i] ^ Y[i]` written directly into Q.
- Final write: `z = Q[i] ^ X[i] ^ Y[i]` — X and Y are re-read from global
  memory to recompute what R used to hold.
- Saves 1 KB of local (off-chip) memory per G_block call, reducing the spill
  footprint from ~2 KB to ~1 KB per call.
- Named registers, macro P and G_MIX, template WITH_XOR, and STEP macros
  retained from LM4.

```cuda
// No R[] array — Q is the only working buffer
for (uint32_t i{ 0u }; i < ARGON2D_LM5_WORDS; ++i)
    Q[i] = X[i] ^ Y[i];

// ...P permutations on Q...

// Recompute X^Y instead of reading from R[]
for (uint32_t i{ 0u }; i < ARGON2D_LM5_WORDS; ++i)
{
    uint64_t z{ Q[i] ^ X[i] ^ Y[i] };
    if (true == WITH_XOR) { z ^= Z[i]; }
    Z[i] = z;
}
```

**Result: ~3.9 MH/s — +30% over LM3/LM4, only -5% vs LM1**

Eliminating R[] is the most effective structural change among LM3–LM7.
Halving the local memory requirement per G_block lets the compiler spill less
and reuse registers more effectively across the 22 inlined calls.

---

### LM6 — `__ldg()` Read-Only Cache + IMAD.WIDE Cast

**Optimisation**: Route X and Y block reads through the read-only L1/texture
cache, and replace `& 0xFFFFFFFFull` with an explicit `(uint64_t)(uint32_t)`
cast to emit IMAD.WIDE.

- `G_MIX_LM6`: `(uint64_t)(uint32_t)(a) * (uint64_t)(uint32_t)(b)` forces a
  32→64-bit widening multiply (IMAD.WIDE) instead of a 64-bit AND followed
  by a 64-bit multiply — fewer instructions on all CUDA architectures.
- Initial XOR: `Q[i] = __ldg(&X[i]) ^ __ldg(&Y[i])` — reads go through the
  read-only data cache, reducing L2 pressure.
- Final write: `z = Q[i] ^ __ldg(&X[i]) ^ __ldg(&Y[i])` — X and Y are read
  a second time via `__ldg()`.
- R[] eliminated like LM5; named registers, macro P, template WITH_XOR,
  STEP macros retained.

```cuda
#define G_MIX_LM6(a, b, c, d)                                                          \
    (a) = (a) + (b) + 2ull * (uint64_t)(uint32_t)(a) * (uint64_t)(uint32_t)(b);        \
    // ...

// Initial XOR — read-only cache path
Q[i] = __ldg(&X[i]) ^ __ldg(&Y[i]);

// Final write — X and Y read again
uint64_t z{ Q[i] ^ __ldg(&X[i]) ^ __ldg(&Y[i]) };
```

**Result: ~2.9 MH/s — SLOWER than LM5 (-25%)**

The double read of X and Y (once for init, once for the final XOR) doubles the
memory traffic for those 128-word arrays compared with LM5's single-read
design. Even with `__ldg()` routing through the texture cache, the extra 1 KB
read per G_block call outweighs the IMAD.WIDE gain. The cast optimisation is
real but secondary.

---

### LM7 — Static Shared Memory, 1 Thread Per Block

**Optimisation**: Move the entire 8-block working set into static shared
memory, eliminating global memory entirely for the Argon2d computation.

- `__shared__ uint64_t smem[8 * 128]` — 8 KiB of static shared memory per
  block, allocated at compile time (no dynamic smem size parameter).
- Kernel launched as `<<<blocks, 1u>>>` — 1 thread per block. The thread
  identity is `blockIdx.x` instead of the usual `blockIdx.x * blockDim.x + threadIdx.x`.
- No `memory` pointer passed to the kernel; `smem` IS the working set.
- Occupancy: 48 KB shared memory per SM ÷ 8 KB per block = 6 concurrent
  blocks per SM.
- G_MIX uses IMAD.WIDE cast (from LM6), named registers, macro P.

```cuda
__global__
void kernel_argon2d_lm7()
{
    __shared__ uint64_t smem[ARGON2D_LM7_Q * ARGON2D_LM7_WORDS];
    uint64_t* const B{ smem };
    uint32_t const threadId{ blockIdx.x };  // 1 thread per block
    // ...
}

// Launched as:
kernel_argon2d_lm7<<<blocks, 1u, 0u, stream>>>();
```

**Result: ~1.15 MH/s — worst variant (-72% vs LM1)**

With 1 thread per block there is no warp-level parallelism. The GPU cannot
hide the latency of sequential G_block operations by switching warps, so
every shared memory access stalls the single active thread. Occupancy of 6
blocks per SM helps slightly but cannot compensate for the complete absence of
thread-level parallelism within each block. For Argon2d with `p=1`, shared
memory offers no advantage: the working set is accessed sequentially and there
is no inter-thread data sharing to exploit.

---

## Optimisation Progression Summary

| Kernel | Key Change                                               | vs LM1  |
|--------|----------------------------------------------------------|---------|
| LM1    | Baseline — runtime loops, function P, bool withXor       | —       |
| LM2    | `#pragma unroll` on all fixed-bound loops in G_block     | **+7%** |
| LM3    | Template WITH_XOR + 22 inlined STEP calls                | -32%    |
| LM4    | Named registers v0–v15 + macro P and G_MIX               | -27%    |
| LM5    | Eliminate R[128] array, recompute XOR in final write     | -5%     |
| LM6    | `__ldg()` for X/Y reads + IMAD.WIDE cast                 | -29%    |
| LM7    | Static shared memory (8 KB), 1 thread per block          | -72%    |

## Key Findings

**`#pragma unroll` (LM2) is the only optimisation that beats the baseline.**
All structural changes — template specialisation, named registers, fill-loop
unrolling, shared memory — produce equal or worse results on this GPU.

**Full fill-loop unrolling (LM3) is severely harmful.** With 22 G_block calls
inlined, each carrying 256 words of local state (R + Q = 2 KB), the compiler
spills heavily to global memory. Loop/branch elimination savings are an order
of magnitude smaller than the spill penalty.

**Eliminating R[] (LM5) is the best structural change**, recovering most of
the LM3 regression by halving local memory pressure per G_block call (2 KB → 1 KB).

**`__ldg()` without R[] (LM6) is counterproductive.** Without R[], X and Y must
be re-read in the final write. Even through the texture cache, this extra 1 KB
read per G_block costs more than IMAD.WIDE saves.

**Shared memory / 1-thread-per-block (LM7) is the worst approach.** Argon2d
with `p=1` is inherently serial within a lane; there is no inter-thread
communication to exploit with shared memory, and 1 thread per block loses
all warp-level latency hiding.

| Bottleneck targeted             | Technique                          | Result   |
|---------------------------------|------------------------------------|----------|
| Loop overhead                   | `#pragma unroll` (LM2)             | **+7%**  |
| `withXor` runtime branch        | Template specialisation (LM3)      | -32%     |
| `v[16]` pointer-array in P      | Named registers + macro P (LM4)    | -27%     |
| R[] local memory pressure       | Eliminate R[] (LM5)                | -5%      |
| Memory bandwidth / multiply     | `__ldg()` + IMAD.WIDE (LM6)        | -29%     |
| Global memory for working set   | Static shared memory (LM7)         | -72%     |
