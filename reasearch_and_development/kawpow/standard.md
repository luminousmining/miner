# KAWPOW Algorithm — Standard Reference

## 1. Overview

**KAWPOW** (KawPoW) is an **ASIC-resistant Proof-of-Work** algorithm derived from ProgPoW. It is designed to favour GPU mining by requiring significant GPU memory bandwidth and complex computation, making it economically impractical to build dedicated hardware (ASICs) for it.

It is used as the consensus mechanism for **Ravencoin (RVN)** and other blockchains.

---

## 2. Core Concepts

### DAG (Directed Acyclic Graph)

The DAG is a large dataset generated from the blockchain's light cache. It grows over time (by epoch). Each mining iteration reads pseudo-random locations from the DAG — the memory access pattern is intentionally irregular to stress GPU memory bandwidth rather than raw compute throughput.

Key constants:

| Constant             | Value      | Description                                  |
|----------------------|------------|----------------------------------------------|
| `LANES`              | 16         | Number of parallel lanes per hash computation |
| `REGS`               | 32         | Number of mix registers per lane             |
| `COUNT_DAG`          | 64         | Number of DAG lookups per hash               |
| `MODULE_CACHE`       | 4096       | Items cached in shared/local memory          |
| `DAG_ITEM_PARENTS`   | 512        | Parents per DAG item (KAWPOW specific)       |

### Hash Functions Used

- **Keccak-f800** — used for seed generation and final digest computation
- **FNV1a** — fast mixing hash used throughout the lane computation
- **KISS99** — pseudo-random number generator driving the math sequence

---

## 3. Algorithm Steps

### Step 1 — Thread and Nonce Initialization

Each GPU thread is assigned a unique **nonce** derived from the kernel's starting nonce and the thread index:

```cpp
uint64_t const nonce = startNonce + thread_id;
```

### Step 2 — Seed Creation

A seed is constructed from the nonce, the block header, and the KAWPOW magic constant (`'r' 'A' 'V' 'E' 'N' 'C' 'O' 'I' 'N' 'K' 'A' 'W' 'P' 'O' 'W'`):

```cpp
create_seed(nonce, state_init, header, &lsb, &msb);
```

Internally:
1. Copy the block header into the Keccak state.
2. Append the nonce.
3. Append the magic constant.
4. Apply `keccak_f800` to produce `lsb` and `msb` (the 64-bit seed split into two 32-bit halves).

### Step 3 — Per-Lane Hash Initialization (`fill_hash`)

For each of the 16 lanes, an initial hash state (32 registers) is computed from the seed and lane index using FNV1a mixing:

```cpp
fill_hash(lane_id, lane_lsb, lane_msb, hash);
```

### Step 4 — DAG Math Loop (`loop_math`)

This is the core of the algorithm and its main source of ASIC resistance. It runs `COUNT_DAG` (64) iterations:

For each iteration:
1. Compute a DAG index from the current hash state and lane ID.
2. Load a 128-byte entry (`uint4`) from the DAG at that index.
3. Run `sequence_math_random` — a sequence of pseudo-random integer operations (add, multiply, XOR, rotate, etc.) driven by KISS99. This sequence changes every block via the `keccak_f800` seed.

```cpp
loop_math(lane_id, dag, hash);
```

The changing math sequence per block is the key feature that makes ProgPoW/KAWPOW difficult to implement in fixed-function hardware.

### Step 5 — Hash Reduction (`reduce_hash`)

After all 64 DAG iterations, the 32-register hash state is reduced to a single 32-bit digest value per lane using FNV1a:

```cpp
reduce_hash(l_id == lane_id, hash, digest);
```

The `digest` array collects one value per lane (16 values total).

### Step 6 — Final Validation (`is_valid`)

The initial seed state and the 16-lane digest are combined through a final `keccak_f800` hash. The 64-bit result is compared against the block difficulty target:

```cpp
uint64_t const bytes = is_valid(state_init, digest);
if (bytes < target) {
    // nonce is valid — block found
}
```

### Step 7 — Result Recording

If the nonce passes validation, it is stored atomically to prevent race conditions between concurrent threads:

```cpp
result->found = true;
uint32_t const index = atomicAdd((uint32_t*)(&result->count), 1);
if (index < 1) {
    result->nonce = nonce;
}
```

---

## 4. Algorithm Summary

| Step | Function         | Description                                          |
|------|------------------|------------------------------------------------------|
| 1    | —                | Assign nonce from thread ID                          |
| 2    | `create_seed`    | Keccak-f800 over header + nonce + magic constant     |
| 3    | `fill_hash`      | Initialize 32 mix registers per lane via FNV1a       |
| 4    | `loop_math`      | 64 × (random DAG load + random math sequence)        |
| 5    | `reduce_hash`    | Reduce 32 registers to one digest value per lane     |
| 6    | `is_valid`       | Final Keccak-f800 and comparison to difficulty target |
| 7    | —                | Atomic store of valid nonce                          |

---

## 5. ASIC Resistance Properties

- **Memory-hard**: 64 random DAG accesses per hash ensure memory bandwidth is the bottleneck, not compute.
- **Compute-variable**: The math sequence between DAG lookups changes every block, preventing static hardware optimisation.
- **Programmatic mix**: KISS99 drives an unpredictable sequence of integer operations (ADD, MUL, XOR, MIN, ROTL, etc.) that a fixed ASIC cannot pre-optimise.
