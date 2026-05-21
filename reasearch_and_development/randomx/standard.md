# RandomX — Simple Developer Guide

## What is RandomX?

RandomX is the **Proof-of-Work algorithm used by Monero** since November 2019. Its core design goal is: make mining as friendly as possible for standard CPUs, and as hostile as possible for specialized hardware (ASICs and FPGAs).

**Simple analogy:** Imagine an exam where every student gets a different set of questions, randomly chosen. A student who memorised answers for common questions would fail — they need a real, general-purpose brain. RandomX works the same way: each nonce causes the GPU/CPU to run a different, random program, so there is no fixed computation to bake into a chip.

---

## Inputs / Output

```
K   — key (0 to 60 bytes). Example: hash of the previous Monero block.
      Changes roughly every 2 days (2048 blocks).
H   — input blob (arbitrary size). Contains the nonce at a fixed offset.

R   — 256-bit (32-byte) result hash.
```

---

## Overview: Three Phases

```
                         K (key, shared, changes rarely)
                         │
                    ┌────▼────────────────────────────────┐
Phase 1a (once/key)│  Cache = Argon2d(K)                  │  256 MiB
                   └────────────────┬─────────────────────┘
                                    │
                    ┌────────────────▼────────────────────┐
Phase 1b (once/key)│  Dataset = SuperscalarHash(Cache)    │  ~2.08 GiB
                   │  ← like building the Ethash DAG      │
                   └────────────────┬─────────────────────┘
                                    │
          H (blob with nonce)       │  dataset shared, read-only
          │                         │
     ┌────▼────────────────────────▼──────────────────────┐
     │  Phase 2 (per nonce)                               │
     │  1. Seed = Blake2b_512(H)                          │
     │  2. Scratchpad = AesGenerator(Seed)  → 2 MiB       │
     │  3. Program   = AesGenerator(...)    → 256 instrs  │
     │  4. VM Loop: 2048 iterations × execute(program)    │
     │     Each iteration reads from dataset + scratchpad │
     └────────────────────────────────────────────────────┘
                         │
                  ┌────▼────────────────────────────────┐
Phase 3           │  R = Blake2b_256(RegisterFile)      │  32 bytes
                  └─────────────────────────────────────┘
```

---

## The Primitives (Building Blocks)

Before diving into the algorithm, here are the primitive functions it relies on.

### Primitive 1: Blake2b

Blake2b is the cryptographic hash function used as the backbone of RandomX.
See [`../blake2b/standard.md`](../blake2b/standard.md) for the full explanation.

Quick summary:
- `Blake2b_512(data)` → 64-byte output
- `Blake2b_256(data)` → 32-byte output
- Deterministic, collision-resistant, fast on CPUs

RandomX uses it to derive seeds, generate programs, and compute the final hash.

---

### Primitive 2: Argon2d

Argon2d fills a large memory buffer in a data-dependent, non-parallelisable way.
See [`../argon2d/standard.md`](../argon2d/standard.md) for the full explanation.

Quick summary:
- Processes memory in passes; each block depends on random previous blocks
- Designed so that skipping blocks (to save RAM) hurts performance badly

RandomX uses Argon2d with these specific parameters to build the 256 MiB **Cache**:

| Parameter   | Value                    |
|-------------|--------------------------|
| Password    | K (the key)              |
| Salt        | `"RandomX\x03"`          |
| Memory      | 262 144 KiB blocks = 256 MiB |
| Iterations  | 3                        |
| Parallelism | 1 lane                   |
| Version     | 0x13                     |
| Type        | Argon2d                  |

Only the filled memory is used — the final Argon2d output hash is discarded.

---

### Primitive 3: BlakeGenerator (Simple PRNG)

A lightweight pseudo-random number generator (PRNG) built on top of Blake2b_512.

**Think of it as:** a bag of random bytes, refilled automatically when empty.

**Internal state:** 64 bytes (a Blake2b_512 digest), plus a read cursor `pos`.

```
Initialize:
  state = Blake2b_512(K)
  pos   = 0

Get 1 byte:
  if pos >= 64 :
    state = Blake2b_512(state)
    pos = 0
  byte = state[pos]
  pos += 1
  return byte

Get 4 bytes (uint32, little-endian):
  if pos > 60 :
    state = Blake2b_512(state)
    pos = 0
  value = state[pos .. pos+3]  (little-endian)
  pos += 4
  return value
```

Used to generate the **SuperscalarHash programs** that build the Dataset.

---

### Primitive 4: AesGenerator1R and AesGenerator4R

Two AES-based stream generators that produce pseudo-random byte streams at very high speed.

**State:** 64 bytes, split into 4 columns of 16 bytes each (`col0, col1, col2, col3`).

Each call produces 64 bytes of output AND updates the state for the next call.

#### AES Background (simplified)

AES operates on a 4×4 grid of bytes (16 bytes = 128 bits). One **round** applies four steps:
1. **SubBytes** — replace each byte using a fixed lookup table (the S-Box)
2. **ShiftRows** — rotate each row left by 0, 1, 2, 3 positions
3. **MixColumns** — linear mix of each column (Galois Field arithmetic)
4. **AddRoundKey** — XOR with a key

**AES encrypt** = SubBytes → ShiftRows → MixColumns → AddRoundKey  
**AES decrypt** = inverse of each step in reverse order

#### AesGenerator1R — Fixed keys, 1 round per column

```
Keys (derived from Blake2b_512("RandomX AesGenerator1R keys")):
  key0 = 53 a5 ac 6d 09 66 71 62  2b 55 b5 db 17 49 f4 b4
  key1 = 07 af 7c 6d 0d 71 6a 84  78 d3 25 17 4e dc a1 0d
  key2 = f1 62 12 3f c6 7e 94 9f  4f 79 c0 f4 45 e3 20 3e
  key3 = 35 81 ef 6a 7c 31 ba b1  88 4c 31 16 54 91 16 49

Each call:
  col0' = AES_decrypt_1_round(col0, key0)
  col1' = AES_encrypt_1_round(col1, key1)
  col2' = AES_decrypt_1_round(col2, key2)
  col3' = AES_encrypt_1_round(col3, key3)
  state = [col0', col1', col2', col3']   → also the 64-byte output
```

Used to initialise the **Scratchpad** (2 MiB of pseudo-random data).

#### AesGenerator4R — 4 rounds per column, two key sets

```
Keys (two sets of 4×16 bytes, derived from Blake2b_512):
  Set A: key0..key3
  Set B: key4..key7

Each call:
  col0' = 4x AES_decrypt(col0, key0, key1, key2, key3)
  col1' = 4x AES_encrypt(col1, key0, key1, key2, key3)
  col2' = 4x AES_decrypt(col2, key4, key5, key6, key7)
  col3' = 4x AES_encrypt(col3, key4, key5, key6, key7)
```

Used to generate the **VM Programs** (256 instructions) for each of the 8 sub-programs.

---

### Primitive 5: AesHash1R (Scratchpad Fingerprint)

Computes a 64-byte hash of the entire 2 MiB Scratchpad, used at the very end.

**Fixed initial state (64 bytes):**
```
state0 = 0d 2c b5 92 de 56 a8 9f  47 db 82 cc ad 3a 98 d7
state1 = 6e 99 8d 33 98 b7 c7 15  5a 12 9e f5 57 80 e7 ac
state2 = 17 00 77 6a d0 c7 62 ae  6b 50 79 50 e4 7c a0 e8
state3 = 0c 24 0a 63 8d 82 ad 07  05 00 a1 79 48 49 99 7e
```

Process the scratchpad in 64-byte chunks:
```
for each 64-byte chunk [k0, k1, k2, k3] :
  state0 = AES_encrypt(state0, k0)
  state1 = AES_decrypt(state1, k1)
  state2 = AES_encrypt(state2, k2)
  state3 = AES_decrypt(state3, k3)
```

Then 2 extra finalisation rounds with fixed keys, producing `state0 || state1 || state2 || state3` = 64 bytes.

---

## Phase 1 — Cache Construction (once per key)

```
Cache = Argon2d(
    password   = K,
    salt       = "RandomX\x03",
    memory     = 262144,    // 262144 KiB = 256 MiB
    iterations = 3,
    lanes      = 1,
    type       = Argon2d
)
```

Result: `Cache[0 .. 268435455]` — 256 MiB of data, `4 194 304` items of 64 bytes.

**This is identical to the Ethash DAG concept**: computed once per key change, shared by all nonces. On Monero, the key is the seed hash derived from the previous block, and it changes every 2048 blocks (~2-3 days).

---

## Phase 2 — VM Execution (per nonce)

This is the core of RandomX. For each nonce, the VM runs independently.

### Step 1: Generate the initial seed

```
seed = Blake2b_512(H)    // H = blob with nonce embedded
```

The 64-byte `seed` drives everything that follows.

### Step 2: Initialise the Scratchpad (2 MiB)

```
gen1_state = seed
Scratchpad[0 .. 2097151] = AesGenerator1R(gen1_state, 2097152 bytes)
gen4_state = gen1_state (after AesGenerator1R finished)
```

Think of the Scratchpad as each nonce's private workspace — 2 MiB of pseudo-random bytes written at the start and used during the VM loop.

### Step 3: VM Registers

```
Integer registers (64-bit, unsigned):
  r0, r1, r2, r3, r4, r5, r6, r7  ← initialised to 0 before the loop

Floating-point registers (pairs of IEEE-754 double):
  f0, f1, f2, f3    ← "additive" — can be positive or negative
  e0, e1, e2, e3    ← "multiplicative" — always ≥ 1 (sign bit forced to +)
  a0, a1, a2, a3    ← constants, read-only, loaded from the program header

Special:
  ma, mx            ← 32-bit Dataset address pointers
  fprc              ← 2-bit IEEE-754 rounding mode (0=nearest, 1=down, 2=up, 3=zero)
```

### Step 4: Generate a Program

For each of the 8 sub-programs, generate 256 instructions using AesGenerator4R:

```
for prog_index in 0..7:
    program_bytes = AesGenerator4R(gen4_state, 2176 bytes)

    // First 128 bytes = program header (initialises a0-a3, ma, mx, fprc mask...)
    // Remaining 2048 bytes = 256 instructions × 8 bytes each
```

**Instruction encoding (8 bytes per instruction):**
```
bits [63:56]  opcode   (8 bits → maps to one of 29 instruction types)
bits [55:48]  dst      (destination register)
bits [47:40]  src      (source register)
bits [39:32]  mod      (modifier flags)
bits [31: 0]  imm32    (32-bit immediate value)
```

### Step 5: The Main Execution Loop (2048 iterations × 8 programs)

The outer structure: run 8 programs, each for 2048 iterations.

For each iteration:

```
// 1. Update scratchpad address pointers
temp   = readReg0 XOR readReg1
spAddr0 ^= (temp & 0xFFFFFFFF)           // low 32 bits
spAddr1 ^= (temp >> 32)                  // high 32 bits

// 2. XOR r0-r7 with 64 bytes from Scratchpad[spAddr0 & L3_MASK]
r0 ^= sp[addr0 + 0];  r1 ^= sp[addr0 + 8];  ...  r7 ^= sp[addr0 + 56]

// 3. Load f0-f3, e0-e3 from 64 bytes at Scratchpad[spAddr1 & L3_MASK]
//    (integers in scratchpad → converted to doubles)

// 4. Execute the 256-instruction program
execute_program()

// 5. Update mx (Dataset address)
mx ^= (readReg2 & 0xFFFFFFFF) XOR (readReg3 & 0xFFFFFFFF)
mx &= 0xFFFFFFC0   // align to 64 bytes

// 6. Read 64 bytes from Dataset (or Cache in light mode) → XOR r0-r7
r0 ^= dataset[ma]; r1 ^= dataset[ma+8]; ... r7 ^= dataset[ma+56]

// 7. Swap mx ↔ ma

// 8. Write r0-r7 back to Scratchpad[spAddr1 & L3_MASK]

// 9. XOR f registers with e registers (bitwise on IEEE-754 representation)
f0 ^= e0;  f1 ^= e1;  f2 ^= e2;  f3 ^= e3

// 10. Write f0-f3 to Scratchpad[spAddr0 & L3_MASK]

// 11. Reset spAddr0 = spAddr1 = 0
```

The Dataset read in step 6 is the key **memory-hard** step: each iteration reads 64 bytes from a data-dependent location in the ~2.08 GiB Dataset (full mode).

---

## The 29 VM Instructions

Divided into 4 categories. The `256 opcode` space is allocated proportionally:

```
Integer instructions  : 120 / 256 opcodes  (46.9%)
Float instructions    :  94 / 256 opcodes  (36.7%)
Control instructions  :  26 / 256 opcodes  (10.2%)
Store instructions    :  16 / 256 opcodes  ( 6.2%)
```

### Integer Instructions (all arithmetic mod 2^64)

| Instruction  | Freq  | Operation                                     |
|--------------|-------|-----------------------------------------------|
| `IADD_RS`    | 16/256 | `dst += src << shift`                        |
| `IADD_M`     |  7/256 | `dst += memory[addr]`                        |
| `ISUB_R`     | 16/256 | `dst -= src`                                 |
| `ISUB_M`     |  7/256 | `dst -= memory[addr]`                        |
| `IMUL_R`     | 16/256 | `dst *= src`                                 |
| `IMUL_M`     |  4/256 | `dst *= memory[addr]`                        |
| `IMULH_R`    |  4/256 | `dst = (dst * src) >> 64`  (unsigned high)   |
| `IMULH_M`    |  1/256 | `dst = (dst * mem) >> 64`  (unsigned high)   |
| `ISMULH_R`   |  4/256 | `dst = (dst * src) >> 64`  (signed high)     |
| `ISMULH_M`   |  1/256 | `dst = (dst * mem) >> 64`  (signed high)     |
| `IMUL_RCP`   |  8/256 | `dst *= floor(2^x / imm32)` (reciprocal mul) |
| `INEG_R`     |  2/256 | `dst = -dst`  (two's complement)             |
| `IXOR_R`     | 15/256 | `dst ^= src`                                 |
| `IXOR_M`     |  5/256 | `dst ^= memory[addr]`                        |
| `IROR_R`     |  8/256 | `dst = rotate_right(dst, src & 63)`          |
| `IROL_R`     |  2/256 | `dst = rotate_left(dst, src & 63)`           |
| `ISWAP_R`    |  4/256 | `swap(dst, src)`                             |

Memory address for `_M` instructions: `(src_register + imm32) & level_mask`
Level selected by `mod.mem` bits (L1=16 KiB, L2=256 KiB, L3=2 MiB).

### Float Instructions (IEEE-754 double, each register holds 2 doubles)

Each float register = `(value_low, value_high)`. Operations apply to **both halves simultaneously**.

| Instruction  | Freq  | Dst | Src | Operation                             |
|--------------|-------|-----|-----|---------------------------------------|
| `FSWAP_R`    |  4/256 | F/E | -  | `(dst[0], dst[1]) = (dst[1], dst[0])` |
| `FADD_R`     | 16/256 | F   | A  | `dst[i] += src[i]`                    |
| `FADD_M`     |  5/256 | F   | R  | `dst[i] += mem_as_double[i]`          |
| `FSUB_R`     | 16/256 | F   | A  | `dst[i] -= src[i]`                    |
| `FSUB_M`     |  5/256 | F   | R  | `dst[i] -= mem_as_double[i]`          |
| `FSCAL_R`    |  6/256 | F   | -  | flip sign + adjust exponent by XOR    |
| `FMUL_R`     | 32/256 | E   | A  | `dst[i] *= src[i]`                    |
| `FDIV_M`     |  4/256 | E   | R  | `dst[i] /= mem_as_double[i]`          |
| `FSQRT_R`    |  6/256 | E   | -  | `dst[i] = sqrt(dst[i])`               |

Converting scratchpad bytes to doubles for `_M` instructions:
- **F group**: interpret 4 bytes as signed int32, cast to double
- **E group**: same, then force sign = positive and set exponent bits

### Control Instructions

**`CFROUND`** (1/256) — Change the IEEE-754 rounding mode:
```
fprc = rotate_right(src_register, imm32) & 3
// 0 = round to nearest, 1 = toward -inf, 2 = toward +inf, 3 = toward zero
```

**`CBRANCH`** (25/256) — Conditional backward jump:
```
b    = mod.cond + 8               // b in [8, 23]
cimm = imm32 | (1 << b)          // force bit b to 1
cimm &= ~(1 << (b-1))            // force bit (b-1) to 0
dst += cimm
if ((dst >> b) & 0xFF) == 0:     // 8 consecutive bits are zero → jump
    goto instruction after last modification of dst
```

Why is this safe? Because cimm always has bit `b` set, adding it repeatedly will eventually make bits `[b..b+7]` all zero — the loop exits in at most 256 iterations.

### Store Instruction

**`ISTORE`** (16/256):
```
address = (dst_register + imm32) & level_mask
scratchpad[address] = src_register  (8 bytes, little-endian)
// Level selected by mod.cond and mod.mem flags
```

---

## Phase 3 — Finalisation

After all 8 programs × 2048 iterations:

```
// Scratchpad fingerprint
A = AesHash1R(Scratchpad)   // 64-byte hash of the entire 2 MiB scratchpad

// Overwrite a0-a3 in the RegisterFile with the fingerprint
RegisterFile[192..255] = A

// Final hash
R = Blake2b_256(RegisterFile)   // RegisterFile = 256 bytes (r0-r7, f0-f3, e0-e3, a0-a3)
```

---

## Light Mode vs Full Mode

| Mode       | Dataset used      | Dataset size | Speed                  |
|------------|-------------------|--------------|------------------------|
| Full mode  | Pre-built Dataset | ~2.08 GiB    | Fast (one lookup/iter) |
| Light mode | Cache directly    | 256 MiB      | Slow (8 lookups/iter per cache item) |

**Full mode (what this benchmark implements):** The Dataset (~2.08 GiB) is built once from the Cache using SuperscalarHash. Each VM iteration reads 64 bytes from a data-dependent address in the Dataset. The Cache (256 MiB) can be freed after the Dataset is built.

**Light mode:** No Dataset. Each Dataset read is replaced by 8 Cache reads combined through SuperscalarHash on the fly. Uses only 256 MiB but is much slower per iteration.

---

## Configuration Constants

| Constant                     | Value          | Meaning                              |
|------------------------------|----------------|--------------------------------------|
| `RANDOMX_CACHE_SIZE`         | 268 435 456    | 256 MiB — Cache from Argon2d         |
| `RANDOMX_SCRATCHPAD_L3`      | 2 097 152      | 2 MiB scratchpad per nonce           |
| `RANDOMX_SCRATCHPAD_L2`      | 262 144        | 256 KiB inner scratchpad             |
| `RANDOMX_SCRATCHPAD_L1`      | 16 384         | 16 KiB inner scratchpad              |
| `RANDOMX_PROGRAM_COUNT`      | 8              | Number of programs per hash          |
| `RANDOMX_PROGRAM_ITERATIONS` | 2048           | VM iterations per program            |
| `RANDOMX_PROGRAM_SIZE`       | 256            | Instructions per program             |
| `RANDOMX_JUMP_OFFSET`        | 8              | CBRANCH: b = mod.cond + 8            |
| `RANDOMX_CACHE_ACCESSES`     | 8              | Cache reads per Dataset item (light) |
| `RANDOMX_DATASET_EXTRA_SIZE` | 33 554 368     | Extra Dataset items                  |
| Argon2d salt                 | `RandomX\x03`  | Fixed Argon2d salt                   |
| Argon2d memory               | 262 144 KiB    | = 256 MiB                            |
| Argon2d iterations           | 3              |                                      |
| Argon2d lanes                | 1              |                                      |

---

## Complete Algorithm Summary

```
INPUT: K (key, shared), H (blob with nonce)

══════════════════════ PHASE 1 (once per K) ═════════════════════════
Cache[256 MiB]   = Argon2d(K, salt="RandomX\x03", mem=256MiB, iter=3)
Dataset[~2.08GiB] = SuperscalarHash(Cache)   // built from Cache; Cache freed after
                    → Dataset stored in GPU global memory, read-only for all threads

════════════════════ PHASE 2 (per nonce) ════════════════════════════
seed = Blake2b_512(H)

  ┌─ Scratchpad init (2 MiB per thread, AesGenerator1R) ──────────┐
  │  Scratchpad = fill 2 MiB with AesGenerator1R(seed)            │
  │  gen4_state = final AesGenerator1R state                      │
  └───────────────────────────────────────────────────────────────┘

  For program_index in [0, 8):
    ┌─ Program generation ───────────────────────────────────────┐
    │  prog_data = AesGenerator4R(gen4_state, 2176 bytes)        │
    │  a0-a3 ← prog_data[0..127] (header)                        │
    │  instructions[256] ← prog_data[128..2175]                  │
    └────────────────────────────────────────────────────────────┘

    ┌─ VM execution loop (2048 iterations) ─────────────────────┐
    │  r[0..7] = 0                                              │
    │  spAddr0 = mx ; spAddr1 = ma                              │
    │  for iter in [0, 2048):                                   │
    │    1. spAddr0 ^= low32(r[readReg0] ^ r[readReg1])         │
    │       spAddr1 ^= high32(r[readReg0] ^ r[readReg1])        │
    │    2. r[0..7] ^= Scratchpad[spAddr0 & L3_MASK .. +64]     │
    │    3. f0-f3, e0-e3 ← Scratchpad[spAddr1 & L3_MASK .. +64] │
    │    4. execute 256 instructions                            │
    │    5. mx ^= low32(readReg2) ^ low32(readReg3)             │
    │       mx &= 0xFFFFFFC0                                    │
    │    6. r[0..7] ^= Dataset[(ma>>6) % DATASET_ITEMS .. +64]  │
    │    7. swap(mx, ma)                                        │
    │    8. Scratchpad[spAddr1 & L3_MASK] = r[0..7]             │
    │    9. f[i] ^= e[i]  (bitwise on IEEE-754 bits)            │
    │   10. Scratchpad[spAddr0 & L3_MASK] = f[0..3]             │
    │   11. spAddr0 = spAddr1 = 0                               │
    └───────────────────────────────────────────────────────────┘

    if program_index < 7:
      gen4_state = Blake2b_512(RegisterFile)  // reseed for next program

═════════════════════ PHASE 3 ═══════════════════════════════════════
A = AesHash1R(Scratchpad)          // 64-byte scratchpad fingerprint
RegisterFile[192..255] = A         // overwrite a0-a3
R = Blake2b_256(RegisterFile)      // final 32-byte output

OUTPUT: R
```
