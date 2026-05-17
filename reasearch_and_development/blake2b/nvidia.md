# Blake2b — NVIDIA CUDA Optimisation Work

## Overview

This document tracks the successive CUDA kernel optimisations developed for Blake2b on NVIDIA GPUs. Each variant (`lm1` through `lm5`) introduces one or more targeted changes. The benchmark runs each kernel 10 times with 128 threads × 8192 blocks.

Each kernel hashes a single 8-byte message block where `m[0] = threadId` and `m[1..15] = 0`, producing a 256-bit digest (Blake2b-256, `digest_size=32`, no key, `fanout=1`, `depth=1`).

Benchmark entry point: [sources/benchmark/nvidia/blake2b.cpp](../../sources/benchmark/nvidia/blake2b.cpp)
Kernel sources: [sources/benchmark/cuda/blake2b/](../../sources/benchmark/cuda/blake2b/)

---

## Benchmark Results

| Kernel | Hashrate (steady state) | Key technique                       |
|--------|------------------------|-------------------------------------|
| LM1    | ~8.5 GH/s              | Baseline                            |
| LM2    | ~8.6 GH/s              | `#pragma unroll`                    |
| LM3    | ~8.6 GH/s              | Named registers + macro G           |
| LM4    | ~45.6 GH/s             | Full manual unroll, no SIGMA table  |
| LM5    | ~8.7 GH/s              | `__ldg()` read-only cache           |

---

## Kernel Variants

### LM1 — Baseline

**Optimisation**: None. This is the correctness reference and performance floor.

- `blake2b_compress_lm1()` is a standalone `__device__ __forceinline__` function.
- Working vector `v[16]` and state `h[8]` are plain local arrays.
- IV and SIGMA are stored in `__constant__` memory.
- SIGMA is accessed at runtime via `s = SIGMA[r % 10]`, then `m[s[i]]` indexing.
- 12 rounds with an explicit `for` loop; no unroll hints.

```cuda
for (uint32_t r{ 0u }; r < 12u; ++r)
{
    uint8_t const* const s{ BLAKE2B_LM1_SIGMA[r % 10u] };
    blake2b_G_lm1(v[0], v[4], v[8],  v[12], m[s[0]],  m[s[1]]);
    // ...
}
for (uint32_t i{ 0u }; i < 8u; ++i)
{
    h[i] ^= v[i] ^ v[i + 8u];
}
```

---

### LM2 — Compiler Unroll Hints

**Optimisation**: Add `#pragma unroll` on all fixed-bound loops.

- Functionally identical to LM1.
- `#pragma unroll` added before: initialisation of `v[]`, the 12-round loop, the finalisation of `h[]`, and the `h[]` init in the kernel.
- **Result**: no measurable gain — NVCC already unrolls constant-bound loops automatically at `-O3`. This confirms that loop unrolling is not a bottleneck.

```cuda
#pragma unroll
for (uint32_t r{ 0u }; r < 12u; ++r)
{
    // ...
}
```

---

### LM3 — Explicit Named Registers + Macro G

**Optimisation**: Replace arrays with named scalar variables to expose register-level values directly to the compiler.

- `h0`–`h7` and `v0`–`v15` declared as individual `uint64_t` local variables instead of arrays.
- G function replaced by a `#define` macro that operates directly on the named variables, eliminating the function call and reference-parameter overhead.
- SIGMA table and round loop retained (`#pragma unroll` on the round loop).
- `m[16]` kept as an array since SIGMA-indexed access requires runtime indexing.

```cuda
#define BLAKE2B_G_LM3(va, vb, vc, vd, x, y) \
    { (va) = (va) + (vb) + (x); (vd) = ror_64((vd) ^ (va), 32u); ... }

uint64_t v0{ h0 };  // explicit register — not v[0]
// ...
BLAKE2B_G_LM3(v0, v4, v8, v12, m[s[0]], m[s[1]]);
```

- **Result**: still ~8.6 GH/s. Named registers help register allocation but the bottleneck remains the SIGMA table indirection (`m[s[i]]` cannot be resolved at compile time).

---

### LM4 — Full Manual Unroll, No SIGMA Table

**Optimisation**: Eliminate the SIGMA table entirely by baking all 96 G-call indices as compile-time constants.

- All 12 rounds hardcoded as straight-line code — no loop, no SIGMA array, no runtime indexing.
- `v0`–`v15` and `h0`–`h7` remain as named scalar variables.
- `m[]` accessed only by literal integer indices, allowing the compiler to treat all message words as constants or keep them in registers.
- The compiler can see the full 96-step computation as a single straight-line sequence and apply maximum instruction scheduling and register allocation.

```cuda
// clang-format off
// Round 0 — sigma[0] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 }
BLAKE2B_G_LM4(v0, v4, v8,  v12, m[ 0], m[ 1]);
BLAKE2B_G_LM4(v1, v5, v9,  v13, m[ 2], m[ 3]);
// ...

// Round 1 — sigma[1] = {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 }
BLAKE2B_G_LM4(v0, v4, v8,  v12, m[14], m[10]);
// ...
```

- **Result**: **~45.6 GH/s** — a **5.3× speedup** over LM1–LM3.
- Root cause: removing the SIGMA indirection (`m[s[i]]`) is the single critical optimisation. Without it, the compiler can resolve all message word accesses statically and eliminate the array load latency entirely.

---

### LM5 — `__ldg()` via Global Memory

**Optimisation**: Move IV and SIGMA from `__constant__` memory to global device memory, accessed through `__ldg()`.

- IV and SIGMA declared as `__device__` (not `__constant__`).
- All reads through `__ldg()`, which routes through the read-only L1 texture cache — a distinct cache path from the constant cache.
- G function kept as a regular `__forceinline__` function; `v[16]` array with `#pragma unroll` (same structure as LM2).

```cuda
// IV and SIGMA in global memory (not __constant__) to use __ldg()
// __ldg() goes through the read-only L1 texture cache, distinct from the constant cache

__device__
uint64_t BLAKE2B_LM5_IV[8] { ... };

// In compress:
v[i + 8] = __ldg(&BLAKE2B_LM5_IV[i]);
blake2b_G_lm5(v[0], v[4], v[8], v[12], m[__ldg(&s[0])], m[__ldg(&s[1])]);
```

- **Result**: ~8.7 GH/s — identical to LM1–LM3. The texture cache does not help here because the bottleneck is the SIGMA indirection, not the cache path used for IV or SIGMA values.

---

## Key Finding

The performance gap between LM3 (~8.6 GH/s) and LM4 (~45.6 GH/s) demonstrates that **eliminating the SIGMA indirection is the only optimisation that matters for Blake2b on GPU**.

With the SIGMA table present, the compiler cannot resolve `m[s[i]]` at compile time: `m[]` must be kept as an array in local memory (spilled or stack-allocated), and each G call involves a runtime index load. This bottleneck persists regardless of loop unrolling, named register variables, or cache path selection.

Without the SIGMA table, all 16 message word accesses become compile-time constants. The compiler can:
- Keep `m[]` values in registers (no array needed)
- See the entire 96-step compression as straight-line code
- Apply aggressive instruction scheduling across all rounds simultaneously

| Bottleneck removed         | Technique                  | Speedup |
|----------------------------|----------------------------|---------|
| Loop overhead              | `#pragma unroll` (LM2)     | ~0%     |
| Array register pressure    | Named variables (LM3)      | ~0%     |
| Constant cache vs L1 cache | `__ldg()` (LM5)            | ~0%     |
| **SIGMA indirection**      | **Hardcoded rounds (LM4)** | **+430%** |
