# Argon2 — Technical Documentation
> **Target audience**: developer, beginner in formal mathematics  
> **Reference implementation**: `crypto/argon2d.py`  
> **Tests**: `crypto/argon2d_test.py`  
> **RFC**: [RFC 9106](https://www.rfc-editor.org/rfc/rfc9106)

---

## Table of Contents

1. [Why Does Argon2 Exist?](#1-why-does-argon2-exist)
2. [The Three Modes: d, i, id](#2-the-three-modes-d-i-id)
3. [Algorithm Overview](#3-algorithm-overview)
4. [Input Parameters](#4-input-parameters)
5. [Step by Step — The Code Explained](#5-step-by-step--the-code-explained)
   - [Step A — Computing Memory Dimensions](#step-a--computing-memory-dimensions)
   - [Step B — Initial Hash H0](#step-b--initial-hash-h0)
   - [Step C — Memory Initialization](#step-c--memory-initialization)
   - [Step D — The Fill Loop](#step-d--the-fill-loop)
   - [Step E — Finalization and Tag](#step-e--finalization-and-tag)
6. [Mathematical Building Blocks (Without the Math)](#6-mathematical-building-blocks-without-the-math)
   - [fBlaMka — Integer Multiplication](#fblaMka--integer-multiplication)
   - [G_mix — The Mixing Function](#g_mix--the-mixing-function)
   - [P — The 16-Word Permutation](#p--the-16-word-permutation)
   - [G(X, Y) — Block Compression](#gx-y--block-compression)
   - [H' — Variable-Length Hashing](#h--variable-length-hashing)
7. [How is the Reference Block Selected?](#7-how-is-the-reference-block-selected)
8. [Argon2i — Independent Addressing](#8-argon2i--independent-addressing)
9. [Public API — How to Use It](#9-public-api--how-to-use-it)
10. [Test Vectors and Validation](#10-test-vectors-and-validation)
11. [Complete Summary Diagram](#11-complete-summary-diagram)
12. [Comparison with Blake2b](#12-comparison-with-blake2b)
13. [RandomX Parameters (Real-World Use Case)](#13-randomx-parameters-real-world-use-case)

---

## 1. Why Does Argon2 Exist?

### The Problem with Classical Hash Functions

SHA-256, Blake2b, MD5 — all these functions compute a hash **very quickly**. That is a quality... but a flaw for password hashing.

An attacker with a GPU can test **billions of passwords per second** against a stolen SHA-256 hash. Argon2 exists to make this **economically impossible**.

### The Solution: Forcing RAM Usage

A GPU may have thousands of cores, but each core shares limited memory. If each hash computation **requires 256 MB of RAM**, an 8 GB GPU can only compute ~32 hashes **simultaneously** instead of millions.

```
SHA-256 :  [CPU/GPU]  →  hash in 1 microsecond, 0 KB of RAM
Argon2d :  [CPU/GPU]  →  hash in 1 second,      256,000 KB of RAM
```

**Argon2 won the Password Hashing Competition (2015)**, standardized in RFC 9106 in 2021. It is used by:
- **Monero / RandomX**: building the 256 MiB cache
- **Bitwarden, 1Password**: hashing master passwords
- **Linux**: authentication via libxcrypt

---

## 2. The Three Modes: d, i, id

| Mode | Type | Memory Addressing | Main Use |
|------|------|-------------------|----------|
| **Argon2d** | 0 | **Data-dependent**: the address of the next block read depends on the data | Cryptocurrencies (RandomX) |
| **Argon2i** | 1 | **Data-independent**: addresses computed in advance | Password hashing |
| **Argon2id** | 2 | **Hybrid**: Argon2i for the 1st sweep, Argon2d afterwards | RFC-recommended mode |

**Why two modes?**

- `Argon2d` is harder to attack via GPU/ASIC (maximum parallelization resistance), but the attacker can monitor memory accesses (*side-channel*) to deduce information about the password.
- `Argon2i` prevents this leak: even without seeing the data, one cannot guess the accesses. But it is slightly weaker against GPUs.
- `Argon2id` combines both: the first pass mixes independently, the subsequent ones benefit from GPU resistance.

In `crypto/argon2d.py`, the choice is made in `_argon2()`:

```python
use_independent_addressing = (
    argon2_type == ARGON2_TYPE_I
    or
    (argon2_type == ARGON2_TYPE_ID and pass_n == 0 and slice_n < 2)
)
```

---

## 3. Algorithm Overview

Here is what Argon2 does, in one diagram:

```
  Inputs
  ──────
  password  +  salt  +  (secret K)  +  (data X)
  parameters: t (passes), m (memory KiB), p (lanes), T (tag size)
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  STEP B: H0 = Blake2b-64( all parameters )              │
  └─────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  STEP C: Allocate m' blocks of 1024 bytes               │
  │          Initialize B[lane][0] and B[lane][1]           │
  │          via the H' function (variable-length hash)     │
  └─────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  STEP D: Main loop (t × 4 × p iterations)               │
  │                                                         │
  │  For each block B[lane][col]:                           │
  │    1. Choose a reference block ref                      │
  │    2. B[lane][col] = G( B[lane][prev], B[ref] )         │
  │       (G = 1024-byte compression)                       │
  └─────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  STEP E: Finalization                                   │
  │    C   = XOR of the last block of each lane             │
  │    tag = H'( C, T )                                     │
  └─────────────────────────────────────────────────────────┘
       │
       ▼
  Output: tag of T bytes
```

---

## 4. Input Parameters

```python
# Excerpt from crypto/argon2d.py — signature of argon2d()
def argon2d(
    password:    bytes,          # P: the secret (password, key...)
    salt:        bytes,          # S: random salt (>= 8 bytes)
    time_cost:   int   = 3,      # t: number of passes over memory
    memory_cost: int   = 65536,  # m: memory in KiB (65536 = 64 MiB)
    parallelism: int   = 1,      # p: number of "lanes" (logical threads)
    hash_len:    int   = 32,     # T: output length in bytes
    secret:      bytes = b'',    # K: optional secret key
    assoc_data:  bytes = b''     # X: optional associated data
) -> bytes:
```

| Parameter | Role | Advice |
|-----------|------|--------|
| `password` | What we want to hash | The user's password |
| `salt` | Makes each hash unique even for the same password | Generate with `os.urandom(16)`, store in plaintext |
| `time_cost` | The larger, the slower | `t=3` is the recommended minimum |
| `memory_cost` | RAM used in KiB | `m=65536` = 64 MiB (recommended minimum) |
| `parallelism` | Parallel lanes | Set to the number of available cores |
| `hash_len` | Result size | 32 bytes (256 bits) in general |
| `secret` | Server-side *pepper* | Optional, rotation possible |
| `assoc_data` | Context tied to the use case | e.g. `b"login"` or `b"api-key"` |

---

## 5. Step by Step — The Code Explained

### Memory Structure

Before going into the steps, we need to visualize how memory is organized.

Argon2 divides memory into a **rectangular grid**:

```
           col 0    col 1    col 2  ...  col q-1
           ──────   ──────   ──────      ──────
Lane 0  │ B[0][0] B[0][1] B[0][2] ... B[0][q-1] │
Lane 1  │ B[1][0] B[1][1] B[1][2] ... B[1][q-1] │
Lane 2  │ B[2][0] B[2][1] B[2][2] ... B[2][q-1] │
...
Lane p-1│ ...                        B[p-1][q-1] │

Each cell B[lane][col] = 1 block of 1024 bytes = 128 64-bit integers
```

Each **lane** is divided into **4 segments** (SYNC_POINTS = 4):

```
Lane i :│ Segment 0 │ Segment 1 │ Segment 2 │ Segment 3 │
        │ s blocks  │ s blocks  │ s blocks  │ s blocks  │
        └───────────────────────────────────────────────┘
                      q = 4 × s blocks total
```

---

### Step A — Computing Memory Dimensions

```python
# crypto/argon2d.py — in _argon2()

# m' must be divisible by 4*p
m_prime     = max(4 * p, (m // (4 * p)) * (4 * p))
q           = m_prime // p        # columns per lane
segment_len = q // SYNC_POINTS    # blocks per segment (SYNC_POINTS = 4)
```

**Why round m'?**

Each lane must have exactly 4 segments of equal size. If `m` does not divide exactly by `4 × p`, it is rounded down.

**Example** with `m=32, p=4` (RFC 9106 vector):

```
m'          = 32           (32 is already divisible by 4×4=16)
q           = 32 / 4 = 8  columns per lane
segment_len = 8  / 4 = 2  blocks per segment
Total memory = 32 blocks × 1024 bytes = 32 KiB ✓
```

---

### Step B — Initial Hash H0

**H0 is the "contract"**: it encodes all inputs and all parameters into 64 bytes. Any change, even of a single bit, produces a completely different H0.

```python
# crypto/argon2d.py — in _argon2()

def len_prefixed(data: bytes) -> bytes:
    return _le32(len(data)) + data  # LE32(length) + data

h0_input = (
    _le32(p)               +   # parallelism (4 bytes)
    _le32(T)               +   # tag length (4 bytes)
    _le32(m)               +   # memory (4 bytes)
    _le32(t)               +   # number of passes (4 bytes)
    _le32(ARGON2_VERSION)  +   # version = 19 (4 bytes)
    _le32(argon2_type)     +   # type 0/1/2 (4 bytes)
    len_prefixed(password) +   # length + password
    len_prefixed(salt)     +   # length + salt
    len_prefixed(secret)   +   # length + secret key
    len_prefixed(assoc_data)   # length + associated data
)
H0 = _blake2b(h0_input, digest_size=64)
```

> **Note on `_le32`**: *LE32* = Little-Endian 32 bits. The integer `p=4` is encoded as `04 00 00 00` (4 bytes, least significant byte first). This is the x86/ARM convention.

**Why include lengths (`len_prefixed`)?**

Without them, `password=b"abc"` with `salt=b""` and `password=b"ab"` with `salt=b"c"` would produce the same H0. The length prefix eliminates this ambiguity.

---

### Step C — Memory Initialization

```python
# crypto/argon2d.py — in _argon2()

B = [[None] * q for _ in range(p)]  # empty grid

for i in range(p):
    # Block 0 of lane i: H'( H0 || 0 || i, T=1024 )
    raw_block_0 = _h_prime(H0 + _le32(0) + _le32(i), BLOCK_SIZE)
    B[i][0] = _bytes_to_words(raw_block_0)

    # Block 1 of lane i: H'( H0 || 1 || i, T=1024 )
    raw_block_1 = _h_prime(H0 + _le32(1) + _le32(i), BLOCK_SIZE)
    B[i][1] = _bytes_to_words(raw_block_1)
```

- `_h_prime(data, T=1024)` produces **1024 bytes** from H0.
- `_bytes_to_words(...)` converts these 1024 bytes into **128 64-bit integers** (the internal representation of a block).
- Each lane has its two initial blocks **different** thanks to the parameter `i`.

---

### Step D — The Fill Loop

This is where the "real" memory consumption happens. The loop fills all the remaining blocks.

```python
# crypto/argon2d.py — in _argon2()

for pass_n in range(t):              # t passes
    for slice_n in range(SYNC_POINTS):   # 4 segments
        for lane in range(p):            # p lanes

            # ... (addressing mode computation)

            for idx in range(segment_len):    # blocks in the segment
                col = slice_n * segment_len + idx

                # Skip the 2 already-initialized blocks
                if pass_n == 0 and slice_n == 0 and idx < 2:
                    continue

                # Previous block (circular)
                prev_col   = (col - 1) % q
                prev_block = B[lane][prev_col]

                # J comes from the previous block (Argon2d) or from a table (Argon2i)
                J  = prev_block[0]           # first 64-bit word
                J1 = J & 0xFFFFFFFF          # low 32 bits  → column selection
                J2 = (J >> 32) & 0xFFFFFFFF  # high 32 bits → lane selection

                ref_lane = J2 % p
                ref_col  = _compute_ref_index(...)

                # Pass 0: simple write
                # Pass > 0: XOR with old value (strengthens diffusion)
                with_xor = (pass_n > 0)
                B[lane][col] = _compress(
                    prev_block,
                    B[ref_lane][ref_col],
                    with_xor,
                    B[lane][col]   # old value if with_xor=True
                )
```

**Why XOR on subsequent passes?**

On pass 0, we fill empty slots. On subsequent passes, we **mix** (XOR) the new result with the old content of the slot. This means that to compute pass N, one needs the result of pass N-1, which forces keeping all the memory.

**Visualization of a pass:**

```
Pass 0:
  B[0][2] = G( B[0][1], B[?][?] )   ← write
  B[0][3] = G( B[0][2], B[?][?] )   ← write
  ...

Pass 1:
  B[0][0] = G( B[0][7], B[?][?] ) XOR B[0][0]   ← XOR!
  B[0][1] = G( B[0][0], B[?][?] ) XOR B[0][1]   ← XOR!
  ...
```

---

### Step E — Finalization and Tag

```python
# crypto/argon2d.py — in _argon2()

# XOR of the last block of each lane
C = B[0][q - 1][:]

for i in range(1, p):
    for w in range(WORDS_PER_BLOCK):    # 128 words
        C[w] = C[w] ^ B[i][q - 1][w]  # XOR word by word

# Convert the 128 words to 1024 bytes
C_bytes = _words_to_bytes(C)

# Apply H' to get exactly T bytes
tag = _h_prime(C_bytes, T)
return tag
```

**Why XOR the last blocks?**

In a real parallel implementation, each lane is processed on a separate thread. The final XOR is the synchronization point: it combines all results into a single block without having to wait for each other during computation.

---

## 6. Mathematical Building Blocks (Without the Math)

### A Block = 128 64-Bit Integers

Argon2 operates on **1024-byte** blocks. Internally, they are viewed as **128 unsigned 64-bit integers** (*uint64*). All operations are performed on these integers.

```python
# crypto/argon2d.py — _bytes_to_words()
# Read 8 bytes at a time to form a 64-bit integer (little-endian)

for i in range(128):            # 128 words
    word = 0
    for byte_pos in range(8):   # 8 bytes per word
        word |= data[i*8 + byte_pos] << (byte_pos * 8)
    words.append(word)
```

**All calculations are modulo 2⁶⁴** (overflows beyond 64 bits are ignored), simulating the natural behavior of C `uint64_t` integers.

---

### fBlaMka — Integer Multiplication

```python
# crypto/argon2d.py — _fBlaMka()

def _fBlaMka(a: int, b: int) -> int:
    a_low = a & 0xFFFFFFFF          # keep the low 32 bits of a
    b_low = b & 0xFFFFFFFF          # keep the low 32 bits of b
    return (a + b + 2 * a_low * b_low) & 0xFFFFFFFFFFFFFFFF
```

**In programmer language:**

```
fBlaMka(a, b) = a + b + 2 × (a mod 2³²) × (b mod 2³²)   (mod 2⁶⁴)
```

This is an **enhanced addition**: in addition to `a + b`, a non-linear term `2 × (low 32 bits of a) × (low 32 bits of b)` is added. This term makes the function impossible to simply "undo", which better resists specialized circuits (ASIC/FPGA).

> **Difference from Blake2b**: Blake2b uses just `a + b + x` (simple addition with external data `x`). Argon2 replaces `x` with the multiplication `2 × a_low × b_low`. This is the key modification.

---

### G_mix — The Mixing Function

```python
# crypto/argon2d.py — _G_mix()

def _G_mix(a, b, c, d):
    # Step 1
    a = _fBlaMka(a, b)
    # Step 2
    d = _rotr64(d ^ a, 32)    # right rotation by 32 bits of (d XOR a)
    # Step 3
    c = _fBlaMka(c, d)
    # Step 4
    b = _rotr64(b ^ c, 24)
    # Step 5
    a = _fBlaMka(a, b)
    # Step 6
    d = _rotr64(d ^ a, 16)
    # Step 7
    c = _fBlaMka(c, d)
    # Step 8
    b = _rotr64(b ^ c, 63)
    return a, b, c, d
```

**What `_rotr64(x, n)` does:**

```
x         = 1000 0001 0010 0011 ...  (64 bits)
rotr64(x, 8): shift 8 positions to the right,
              the 8 bits exiting on the right "come back" on the left.
```

In Python: `((x >> n) | (x << (64 - n))) & 0xFFFFFFFFFFFFFFFF`

`_G_mix` takes **4 64-bit integers** (`a, b, c, d`) and **mixes them mutually** in 8 steps. After these 8 steps, each bit of `a` depends on every bit of the 4 inputs. This is a local "avalanche".

---

### P — The 16-Word Permutation

```python
# crypto/argon2d.py — _apply_P()

# The 16 words are viewed as a 4×4 square:
#    v[0]   v[1]   v[2]   v[3]
#    v[4]   v[5]   v[6]   v[7]
#    v[8]   v[9]  v[10]  v[11]
#   v[12]  v[13]  v[14]  v[15]

# 4 G_mix calls on columns:
v[0], v[4], v[8],  v[12] = _G_mix(v[0], v[4], v[8],  v[12])  # column 0
v[1], v[5], v[9],  v[13] = _G_mix(v[1], v[5], v[9],  v[13])  # column 1
v[2], v[6], v[10], v[14] = _G_mix(v[2], v[6], v[10], v[14])  # column 2
v[3], v[7], v[11], v[15] = _G_mix(v[3], v[7], v[11], v[15])  # column 3

# 4 G_mix calls on diagonals:
v[0], v[5], v[10], v[15] = _G_mix(v[0], v[5], v[10], v[15])  # diag. 0
v[1], v[6], v[11], v[12] = _G_mix(v[1], v[6], v[11], v[12])  # diag. 1
v[2], v[7], v[8],  v[13] = _G_mix(v[2], v[7], v[8],  v[13])  # diag. 2
v[3], v[4], v[9],  v[14] = _G_mix(v[3], v[4], v[9],  v[14])  # diag. 3
```

**Visualization of columns and diagonals:**

```
  Columns (vertical)         Diagonals
  ─────────────────────      ──────────────────────
  v[0]  .    .    .          v[0]  .    .    .
  v[4]  .    .    .           .   v[5]  .    .
  v[8]  .    .    .           .    .   v[10]  .
  v[12] .    .    .           .    .    .   v[15]
```

This **column then diagonal** structure is exactly that of Blake2b (which borrowed it from ChaCha20). It ensures that after P, all 16 words depend on each other.

---

### G(X, Y) — Block Compression

```python
# crypto/argon2d.py — _compress()
# Takes two blocks of 128 words, produces a block of 128 words.

# 1. R = X XOR Y  (word by word)
for i in range(128):
    R[i] = X[i] ^ Y[i]

# 2. Q = copy of R
Q = R[:]

# 3. ROW permutations (8 rows of 16 consecutive words)
for l in range(8):
    v = [Q[l*16 + k] for k in range(16)]  # extract
    _apply_P(v)                             # mix
    for k in range(16): Q[l*16 + k] = v[k] # put back

# 4. COLUMN permutations (8 columns of 16 interleaved words)
for l in range(8):
    indices = [2*l + k*16 + offset for k in range(8) for offset in (0, 1)]
    v = [Q[idx] for idx in indices]
    _apply_P(v)
    for i, idx in enumerate(indices): Q[idx] = v[i]

# 5. Z = Q XOR R
for i in range(128):
    Z[i] = Q[i] ^ R[i]
```

**Block viewed as an 8 × 16 matrix:**

```
  Row 0  : words [  0,  1,  2, ..., 15 ]  ← P applied horizontally
  Row 1  : words [ 16, 17, 18, ..., 31 ]
  ...
  Row 7  : words [112,113, ...,     127]

  Column 0: words [  0,  1, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97,112,113]
  Column 1: words [  2,  3, 18, 19, 34, 35, ...]
  ...                                        ↑ P applied vertically
```

The final operation `Z = Q XOR R` (step 5) is a design detail that makes G **involutive** — a property that facilitates security analysis.

---

### H' — Variable-Length Hashing

Blake2b produces at most **64 bytes**. For blocks (1024 bytes) and tags of arbitrary size, Argon2 defines `H'`:

```python
# crypto/argon2d.py — _h_prime()

def _h_prime(data: bytes, T: int) -> bytes:

    if T <= 64:
        # Simple case: a single Blake2b call
        return blake2b( LE32(T) + data, digest_size=T )

    else:
        # Long case: chain of calls, 32 bytes collected each time
        r    = ceil(T/32) - 2
        A_1  = blake2b( LE32(T) + data, digest_size=64 )  ← first hash
        A_2  = blake2b( A_1,            digest_size=64 )
        ...
        A_r  = blake2b( A_{r-1},        digest_size=64 )
        A_r+1= blake2b( A_r,            digest_size=T-32*r )

        result = A_1[0:32] + A_2[0:32] + ... + A_r[0:32] + A_r+1
```

**Example for T = 1024 (one Argon2 block):**

```
r    = ceil(1024/32) - 2 = 32 - 2 = 30
A_1  = Blake2b(LE32(1024) + H0 + ...,  64 bytes)  → take first 32
A_2  = Blake2b(A_1,                    64 bytes)  → take first 32
...
A_30 = Blake2b(A_29,                   64 bytes)  → take first 32
A_31 = Blake2b(A_30,  T - 32×30 = 64 bytes)       → take all 64

Total: 30 × 32 + 64 = 960 + 64 = 1024 bytes ✓
```

---

## 7. How is the Reference Block Selected?

For each new block `B[lane][col]`, Argon2 chooses a reference block from among the **already-computed** blocks. This choice is at the heart of GPU resistance.

### Extracting J1 and J2

```python
# For Argon2d: J comes from the first word of the previous block
J  = prev_block[0]           # 64 bits
J1 = J & 0xFFFFFFFF          # low 32 bits  → column index
J2 = (J >> 32) & 0xFFFFFFFF  # high 32 bits → lane index
```

### Choosing the Reference Lane

```python
if pass_n == 0 and slice_n == 0:
    ref_lane = lane     # 1st pass, 1st segment: stay in the same lane
else:
    ref_lane = J2 % p   # any lane
```

**Why stay in the same lane at the beginning?**

At the start of pass 0, the other lanes are not yet filled. We can only reference what exists.

### Choosing the Column via the Phi Formula

```python
# crypto/argon2d.py — _compute_ref_index()

x         = J1
step_a    = (x * x) >> 32           # J1² / 2³²
step_b    = (ref_area * step_a) >> 32
relative_pos = ref_area - 1 - step_b
```

**What this formula does:**

It converts J1 (a pseudo-random value) into a **non-uniform index**: recent blocks are **more likely to be chosen**. This is analogous to an inverted `x²` distribution.

```
The closer J1 is to 0    → a recent block is chosen
The closer J1 is to 2³² → an older block is chosen
```

**Why?**

Recent blocks are more often in the CPU cache. Favoring recent blocks makes the algorithm harder to run in batches (an attacker who tries to skip blocks loses their cache data).

### Available Reference Area

The `ref_area` formula follows precise rules depending on the position in the loop:

| Situation | Reference Area |
|-----------|----------------|
| Pass 0, segment 0, same lane | Only blocks `[0 .. idx-1]` |
| Pass 0, segment>0, same lane | Blocks from the start up to `idx-1` |
| Pass 0, segment>0, other lane | Complete previous segments |
| Pass>0, same lane | Entire lane except the current segment |
| Pass>0, other lane | Entire lane except the current segment |

---

## 8. Argon2i — Independent Addressing

For Argon2i, J1 and J2 do **not** come from the previous block. They are generated in advance via compression:

```python
# crypto/argon2d.py — _generate_addresses()

# Input block encoding the position
input_block = [0] * 128
input_block[0] = pass_n       # pass
input_block[1] = lane         # lane
input_block[2] = slice_n      # segment
input_block[3] = m_prime      # total memory size
input_block[4] = t            # number of passes
input_block[5] = argon2_type  # type (1=i, 2=id)

# For each batch of 128 values:
counter += 1
input_block[6] = counter

zeros      = [0] * 128
tmp        = G( zeros, input_block )   # first pass
addr_block = G( zeros, tmp )           # second pass

J[k] = addr_block[k]   # 128 pseudo-random values
```

**Why two passes of G?**

A single pass of G is not sufficiently "diffused" — the inputs and output would have too direct a relationship. The double pass guarantees that no bit of the position leaks into J.

**Argon2id:**

```python
use_independent_addressing = (
    argon2_type == ARGON2_TYPE_I
    or
    (argon2_type == ARGON2_TYPE_ID and pass_n == 0 and slice_n < 2)
)
```

For `Argon2id`, only the **first 2 segments of the 1st pass** use independent addressing. This is sufficient to eliminate the most exploitable leaks, while preserving the GPU resistance of subsequent passes.

---

## 9. Public API — How to Use It

### Minimal Usage

```python
from crypto.argon2d import argon2d, argon2i, argon2id
import os

# Generate a random salt (to be stored with the hash)
salt = os.urandom(16)

# Hash a password
tag = argon2d(
    password    = b"my_password",
    salt        = salt,
    time_cost   = 3,        # 3 passes
    memory_cost = 65536,    # 64 MiB
    parallelism = 4,        # 4 lanes
    hash_len    = 32        # 32 bytes of output
)

print(tag.hex())  # e.g.: 9e34c31a47866ce0...
```

### Comparing the Three Modes

```python
pwd  = b"password"
salt = b"saltsalt"
kw   = dict(time_cost=1, memory_cost=8, parallelism=1, hash_len=32)

h_d  = argon2d( pwd, salt, **kw)
h_i  = argon2i( pwd, salt, **kw)
h_id = argon2id(pwd, salt, **kw)

# The three produce DIFFERENT results (the type is included in H0)
assert h_d != h_i != h_id
```

### With Secret Key and Associated Data

```python
# Secret key (K): a "pepper" stored server-side, never in the database
pepper = b"server-side-pepper-key"

# Associated data (X): context of the use case
context = b"login-web-app-v2"

tag = argon2id(
    password   = b"password123",
    salt       = os.urandom(16),
    secret     = pepper,
    assoc_data = context,
    time_cost   = 3,
    memory_cost = 131072,   # 128 MiB
    parallelism = 4,
    hash_len    = 32
)
```

---

## 10. Test Vectors and Validation

The file `crypto/argon2d_test.py` performs two types of verification:

### Part 1 — Comparison Against argon2-cffi

```
argon2-cffi = the official C library, wrapped for Python
Our from-scratch implementation must produce exactly the same bytes.
```

```
python crypto/argon2d_test.py

PART 1: Comparison against argon2-cffi (official C reference)
────────────────────────────────────────────────────────────────────
Minimal (t=1, m=8, p=1, 32B)      Argon2d   PASS
Minimal (t=1, m=8, p=1, 32B)      Argon2i   PASS
Minimal (t=1, m=8, p=1, 32B)      Argon2id  PASS
...
Overall results: 80/80 PASS  |  0 FAIL
```

### Part 2 — RFC 9106 Vectors

The official RFC vectors include a secret key K and associated data X:

| Mode | Password | Salt | K | X | t | m | p | Expected Tag |
|------|----------|------|---|---|---|---|---|--------------|
| Argon2d | `01`×32 | `02`×16 | `03`×8 | `04`×12 | 3 | 32 | 4 | `512b391b...` |
| Argon2id | `01`×32 | `02`×16 | `03`×8 | `04`×12 | 3 | 32 | 4 | `0d640df5...` |

```python
# Reproduce the RFC 9106 vector Annex B.1
from crypto.argon2d import argon2d

tag = argon2d(
    password   = bytes([0x01] * 32),
    salt       = bytes([0x02] * 16),
    secret     = bytes([0x03] * 8),
    assoc_data = bytes([0x04] * 12),
    time_cost   = 3,
    memory_cost = 32,
    parallelism = 4,
    hash_len    = 32
)
assert tag.hex() == "512b391b6f1162975371d30919734294f868e3be3984f3c1a13a4db9fabe4acb"
```

---

## 11. Complete Summary Diagram

```
  INPUTS
  ──────
  password P, salt S, secret K, assoc_data X
  time_cost t, memory_cost m, parallelism p, hash_len T
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  H0 = Blake2b-64(LE32(p)||LE32(T)||LE32(m)||LE32(t)||           │
  │                  LE32(19)||LE32(type)||                         │
  │                  LE32(|P|)||P||LE32(|S|)||S||                   │
  │                  LE32(|K|)||K||LE32(|X|)||X)                    │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
        ┌────────────────────────┼──────────────────────────────────┐
        │   Memory grid B        │                                  │
        │   ─────────────────    │                                  │
        │   For each lane i:     │                                  │
        │     B[i][0] = H'(H0 + LE32(0) + LE32(i), 1024)            │
        │     B[i][1] = H'(H0 + LE32(1) + LE32(i), 1024)            │
        └────────────────────────┬──────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Fill loop             │
                    │   pass=0..t-1           │
                    │   slice=0..3            │
                    │   lane=0..p-1           │
                    │   idx=0..s-1            │
                    │                         │
                    │   J = prev_block[0] (d) │
                    │   J = addr_table[idx](i)│
                    │   J1=J&0xFFFFFFFF       │
                    │   J2=J>>32              │
                    │                         │
                    │   ref_lane = J2 % p     │
                    │   ref_col  = phi(J1)    │
                    │                         │
                    │   B[lane][col] =        │
                    │     G(prev, B[ref_lane] │
                    │         [ref_col])      │
                    │   (XOR if pass>0)       │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼───────────────────────────────┐
        │   Finalization         │                               │
        │   ──────────           │                               │
        │   C = XOR(B[0][q-1], B[1][q-1], ..., B[p-1][q-1])      │
        │   tag = H'(C, T)                                       │
        └────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
                          TAG (T bytes)


  Functions used
  ───────────────────
  H'(data, T):
    If T<=64 → Blake2b(LE32(T)||data, T)
    Else     → chain of Blake2b, 32 bytes/call

  G(X, Y):
    R = X XOR Y
    Q = R  (copy)
    P(rows of Q)    × 8   [P = 4 G_mix columns + 4 G_mix diagonals]
    P(columns of Q) × 8
    Z = Q XOR R
    [if with_xor: Z XOR= C_old]

  G_mix(a,b,c,d):
    a = fBlaMka(a,b)  ;  d = rotr64(d^a, 32)
    c = fBlaMka(c,d)  ;  b = rotr64(b^c, 24)
    a = fBlaMka(a,b)  ;  d = rotr64(d^a, 16)
    c = fBlaMka(c,d)  ;  b = rotr64(b^c, 63)

  fBlaMka(a, b) = a + b + 2×(a mod 2³²)×(b mod 2³²)  mod 2⁶⁴
```

---

## 12. Comparison with Blake2b

Blake2b is used **inside** Argon2 (for H0 and H'). Here are the differences:

| | Blake2b | Argon2 |
|--|---------|--------|
| **Type** | Hash function | KDF (key derivation function) |
| **RAM** | ~few KB (internal state) | `m` KiB (configurable) |
| **Configurable slowness** | No | Yes (t passes) |
| **Mixing** | `a + b + x` (addition + message) | `a + b + 2·a_low·b_low` (multiplication) |
| **Usage** | File hashing, signatures | Password hashing, PoW |
| **Parallelism** | No | Yes (lanes) |

---

## 13. RandomX Parameters (Real-World Use Case)

RandomX (Monero) uses Argon2d to build its **256 MiB Cache**:

```python
# RandomX parameters defined in randomx.h
RANDOMX_ARGON_MEMORY     = 262144    # 262144 KiB = 256 MiB
RANDOMX_ARGON_ITERATIONS = 3        # 3 passes
RANDOMX_ARGON_LANES      = 1        # 1 lane
RANDOMX_ARGON_SALT       = b"RandomX\x03"  # fixed salt

from crypto.argon2d import argon2d

cache = argon2d(
    password    = seed_hash,              # Monero block hash (32 bytes)
    salt        = b"RandomX\x03",
    time_cost   = 3,
    memory_cost = 262144,               # 256 MiB!
    parallelism = 1,
    hash_len    = 32,                   # only the tag, not the entire memory
)
# Note: in practice, RandomX uses the ENTIRE internal memory array B,
# not just the final tag. The 256 MiB cache = the complete B array.
```

> **Note**: The Python implementation is algorithmically correct but far too slow for 256 MiB. In practice, RandomX uses the C library (`librandomx`) via ctypes or official Python bindings. The `crypto/randomx.py` script uses a Blake2b approximation for demonstration, clearly documented.
