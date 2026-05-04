# ETHASH — NVIDIA CUDA Optimisation Work

## Overview

This document tracks the CUDA kernel optimisations tested for ETHASH on NVIDIA GPUs. Each variant (`lm1` through `lm6`) introduces one or more targeted changes against the baseline. The benchmark runs each kernel 10 times with 128 threads × 8192 blocks on a **RTX 4070 Ti**.

Benchmark entry point: [sources/benchmark/nvidia/ethash.cpp](../../sources/benchmark/nvidia/ethash.cpp)
Kernel sources: [sources/benchmark/cuda/ethash/](../../sources/benchmark/cuda/ethash/)

---

## Reference Implementations

| Label       | Description                                             | MH/s   |
|-------------|---------------------------------------------------------|--------|
| `base`      | Naive reference — no warp-level parallelism             | ~13.78 |
| `ethminer`  | Open-source ethminer kernel (uint2 Keccak + lop3 PTX + `_PARALLEL_HASH=4`) | ~51.32 |

---

## Kernel Variants

### LM1 — Baseline

**Optimisation**: None. Starting point for the LM series.

- One nonce per thread.
- Keccak-f[1600] state as `uint64_t[25]`.
- 8 threads collaborate on one nonce: `THREADS_PER_HASH = 8`.
- Mix hash processed serially: 8 lanes × 64 DAG accesses, one lane at a time.
- DAG index computed with native `%` operator.
- DAG pointer passed as kernel parameter with `__restrict__`.

**Result: ~50.91 MH/s**

---

### LM2 — Fast Modulo (`FastDivisor`)

**Optimisation**: Replace the `%` operator with a precomputed magic-number divisor.

```cuda
// Before (lm1)
start_index %= d_dag_number_item;

// After (lm2)
start_index = fast_mod(d_dag_divisor, start_index);
// → __umulhi(value, magic) >> shift  +  correction loop
```

The `FastDivisor` struct is precomputed on the host and uploaded to `__constant__` memory. At runtime, the division is replaced by a high-part multiply and shift.

**Result: ~48.75 MH/s — SLOWER than LM1 (-2.16 MH/s)**

The `fast_mod` implementation includes a `while (r >= d)` correction loop that introduces a conditional branch executed 64 times per nonce. On Ada Lovelace, the hardware integer divider is fast enough that the branch overhead of the correction outweighs the gain from avoiding division. The compiler also already optimises `%` with a constant-range divisor.

---

### LM3 — Fast Modulo (isolated, clean variant)

**Optimisation**: Same `fast_mod` idea as LM2, rebuilt from LM1 as a clean isolated measurement.

Functionally identical to LM2. Created to confirm that the LM2 result was not a measurement artefact.

**Result: ~48.70 MH/s — SLOWER than LM1 (-2.21 MH/s)**

Confirms that `fast_mod` is consistently counterproductive on this architecture. Do not use for DAG index computation.

---

### LM4 — Parallel Hash (`_PARALLEL_HASH = 4`)

**Optimisation**: Process 4 nonces simultaneously per 8-thread group instead of 1.

```cuda
constexpr uint32_t PARALLEL_HASH{ 4u };

// 2 passes of 4 instead of 8 serial passes of 1
for (uint32_t lane_base{ 0u }; lane_base < THREADS_PER_HASH; lane_base += PARALLEL_HASH)
{
    uint4 mix[PARALLEL_HASH];  // 4 independent mix states

    for (uint32_t a{ 0u }; a < ACCESSES; a += 4u)
    {
        for (uint32_t b{ 0u }; b < 4u; ++b)
        {
            for (uint32_t p{ 0u }; p < PARALLEL_HASH; ++p)
            {
                // 4 independent DAG loads issued per inner iteration
                start_index = fnv1(...) % d_dag_number_item;
                start_index = reg_load(start_index, t, THREADS_PER_HASH);
                fnv1(mix[p], dag[start_index + thread_lane_id]);
            }
        }
    }
}
```

Having 4 independent memory loads in flight simultaneously allows the GPU memory scheduler to better hide DRAM latency.

**Result: ~51.05 MH/s — no significant gain (+0.14 MH/s)**

The RTX 4070 Ti memory subsystem already pipelines DAG accesses efficiently with the serial approach. The extra register pressure from holding 4 `mix[]` arrays simultaneously cancels out the latency hiding benefit. Ethminer achieves ~51.32 MH/s with this technique likely because its `uint2` Keccak produces lighter overall register usage.

---

### LM5 — Keccak `uint2` State + PTX `lop3`

**Optimisation**: Replace `uint64_t[25]` Keccak state with `uint2[12]` (ethminer style), enabling PTX `lop3.b32` instructions.

```cuda
// Before (lm1): uint64_t operations, no lop3
uint64_t state[25];
keccak_f1600_round(state, i);  // multiple 64-bit XOR/rotate ops per step

// After (lm5): uint2 state, PTX lop3
uint2 keccak_state[12];
ethash_keccak_f1600_init(keccak_state);

// Inside the ethminer Keccak: theta uses lop3 (3-way XOR in 1 instruction)
// lop3.b32 %0, %2, %3, %4, 0x96;   → a^b^c in 1 PTX instruction
// Chi uses lop3 (a^(~b&c) in 1 instruction)
// lop3.b32 %0, %2, %3, %4, 0xD2;   → a^(~b&c) in 1 PTX instruction
```

Because `lop3.b32` only exists for 32-bit operands, splitting each `uint64_t` into two `uint32_t` values (as `uint2`) allows exploiting this instruction for every Keccak step. The `ethash_keccak_f1600_final` function returns the result directly as `uint64_t` without an explicit final Keccak state array.

The mix hash section is unchanged from LM1 but receives the seed via a `uint2→uint4` conversion.

**Result: ~50.73 MH/s — slightly slower than LM1 (-0.18 MH/s)**

NVCC compiles the `uint64_t` Keccak very efficiently on Ada Lovelace. The `uint2↔uint64_t` seed conversion overhead is small but measurable. The `lop3` benefit exists but does not fully compensate on this microarchitecture.

---

### LM6 — All Combined (`fast_mod` + `PARALLEL_HASH=4` + Keccak `uint2`)

**Optimisation**: Combination of LM3 + LM4 + LM5 simultaneously.

- Keccak-f[1600] as `uint2[12]` with `lop3` PTX (from LM5).
- Mix hash with `PARALLEL_HASH = 4` (from LM4).
- DAG index via `fast_mod` (from LM3).

**Result: ~48.83 MH/s — SLOWER than LM1 (-2.08 MH/s)**

The result is dominated by the `fast_mod` regression. Combining all three changes produces no synergy — the independent gains (LM4, LM5) are marginal and the `fast_mod` penalty carries through.

---

## Result Summary

Test hardware: **NVIDIA GeForce RTX 4070 Ti** (Ada Lovelace, sm_89)
Configuration: 128 threads × 8192 blocks × 10 iterations

| Kernel    | MH/s   | vs LM1   | Key Change                                        |
|-----------|--------|----------|---------------------------------------------------|
| `base`    | ~13.78 | —        | Naive reference                                   |
| `ethminer`| ~51.32 | +0.41    | uint2 Keccak + lop3 + `_PARALLEL_HASH=4`          |
| `lm1`     | ~50.91 | baseline | uint64_t Keccak, `%`, serial mix hash             |
| `lm2`     | ~48.75 | -2.16    | `fast_mod` (`FastDivisor`)                        |
| `lm3`     | ~48.70 | -2.21    | `fast_mod` (clean isolated measurement)           |
| `lm4`     | ~51.05 | +0.14    | `_PARALLEL_HASH=4` (4 nonces per 8-thread group)  |
| `lm5`     | ~50.73 | -0.18    | Keccak `uint2` + `lop3` PTX                       |
| `lm6`     | ~48.83 | -2.08    | LM3 + LM4 + LM5 combined                         |

## Conclusions

- **`fast_mod` is harmful** for DAG index computation on Ada Lovelace. The `while` correction branch executed 64×/nonce costs more than the division saves. Do not use.
- **`_PARALLEL_HASH=4`** is neutral on this GPU (+0.14 MH/s). The Ada Lovelace memory scheduler already pipelines accesses efficiently.
- **Keccak `uint2`+`lop3`** provides no measurable benefit here (-0.18 MH/s). NVCC optimises the `uint64_t` path sufficiently.
- **LM1 is near-optimal** for this architecture. The 0.41 MH/s gap vs ethminer (~0.8%) is likely from its manually unrolled sparse round 0 of Keccak-init (not yet tested in isolation).
- Next investigation: isolate the round-0 sparse optimisation from ethminer's `ethash_keccak_f1600_init`.
