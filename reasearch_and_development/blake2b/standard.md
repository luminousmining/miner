# Blake2b — Technical Documentation

> **Who is this document for?**
> You are an expert programmer, but a beginner in mathematics.
> This document explains Blake2b step by step, in programmer's language.
> Every mathematical concept is immediately translated into code.
>
> **Reference files:**
> - Implementation : `crypto/blake2b.py`
> - Tests          : `crypto/blake2b_test.py`
> - Specification  : [RFC 7693](https://www.rfc-editor.org/rfc/rfc7693)

---

## Table of Contents

1. [What is Blake2b?](#1-what-is-blake2b)
2. [What Blake2b guarantees](#2-what-blake2b-guarantees)
3. [Algorithm overview](#3-algorithm-overview)
4. [Constants: IV and SIGMA](#4-constants-iv-and-sigma)
5. [Data representation: bytes and 64-bit words](#5-data-representation-bytes-and-64-bit-words)
6. [Primitive 1: bit rotation (rotr64)](#6-primitive-1-bit-rotation-rotr64)
7. [Primitive 2: the G function (the heart of Blake2b)](#7-primitive-2-the-g-function-the-heart-of-blake2b)
8. [Compression F: 12 rounds of G](#8-compression-f-12-rounds-of-g)
9. [Main algorithm: blake2b()](#9-main-algorithm-blake2b)
10. [Keyed mode (MAC)](#10-keyed-mode-mac)
11. [Why it is secure: intuition without maths](#11-why-it-is-secure-intuition-without-maths)
12. [Blake2b vs Blake2s vs SHA-512](#12-blake2b-vs-blake2s-vs-sha-512)
13. [Test vectors](#13-test-vectors)
14. [Complete summary diagram](#14-complete-summary-diagram)

---

## 1. What is Blake2b?

Blake2b is a **cryptographic hash function**.

In practice: you give it any amount of data (0 bytes, 1 MB, 1 GB), and it returns a fixed-size block (between 1 and 64 bytes) called a **hash**, **fingerprint** or **digest**.

```python
from crypto.blake2b import blake2b

# Any input
hash = blake2b(b"hello world", digest_size=32)
# => always 32 bytes

hash = blake2b(b"", digest_size=64)
# => 64 bytes even for an empty input
```

**Blake2b is the 64-bit version** of the Blake2 family. There is also:
- **Blake2s**: 32-bit version, for embedded systems
- **Blake2bp / Blake2sp**: parallel versions (multi-thread)

In RandomX and in `crypto/blake2b.py`, only Blake2b (64-bit) is used.

---

## 2. What Blake2b guarantees

Four fundamental properties:

| Property | What it means in practice |
|---|---|
| **Deterministic** | Same input = same hash, always |
| **Avalanche effect** | Changing 1 bit in the input changes ~50% of the hash bits |
| **Non-reversible** | Impossible to recover the input from the hash |
| **Collision-resistant** | Impossible to find two different inputs with the same hash |

These properties are **not mathematical axioms** that are formally proved. They are **empirical observations** validated by the cryptographic community since 2012.

---

## 3. Algorithm overview

Blake2b processes data **in 128-byte blocks**, one block at a time. It maintains an **internal state** `h` of 8 64-bit words (= 64 bytes total).

```
Input : [  data  ]  (arbitrary size)
              |
              v
   Split into 128-byte blocks
   (the last block is zero-padded if incomplete)
              |
              v
   +-----------+     +-----------+     +-----------+
   |  Block 1  | --> |  Block 2  | --> |   Last    |
   | compress  |     | compress  |     | compress  |
   +-----------+     +-----------+     +-----------+
          ^                                  |
          |                                  v
   initial h (8 x 64 bits)         final h (8 x 64 bits)
   = configured IV                           |
                                             v
                                  Serialize to bytes
                                  Truncate to digest_size
                                             |
                                             v
                                       [ HASH ]
```

The compression function mixes **one message block** into **state h**.
After all blocks, state `h` contains the hash.

---

## 4. Constants: IV and SIGMA

### 4.1 The Initialization Vector IV

`IV` is a list of 8 64-bit integers. These values are not magic: they come from the fractional parts of the square roots of the first 8 prime numbers.

```python
# In crypto/blake2b.py
IV = [
    0x6A09E667F3BCC908,   # sqrt(2)  -> 0.41421356... -> take the bits of the fractional part
    0xBB67AE8584CAA73B,   # sqrt(3)
    0x3C6EF372FE94F82B,   # sqrt(5)
    0xA54FF53A5F1D36F1,   # sqrt(7)
    0x510E527FADE682D1,   # sqrt(11)
    0x9B05688C2B3E6C1F,   # sqrt(13)
    0x1F83D9ABFB41BD6B,   # sqrt(17)
    0x5BE0CD19137E2179,   # sqrt(19)
]
```

> **Why square roots?**
> These values are "neutral" — they hide no backdoor because their origin is public and verifiable. It is a convention established since SHA-2. They are called "nothing-up-my-sleeve numbers".

These same constants are used by SHA-512. Blake2b inherited them.

### 4.2 The SIGMA table (message permutation)

`SIGMA` is a table of 10 rows x 16 indices. It defines **in what order** the 16 words of the message block are used at each round.

```python
# In crypto/blake2b.py
SIGMA = [
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],  # round 0: natural order
    [14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3],  # round 1: shuffled
    [11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4],  # round 2: another shuffle
    # ... 7 more rows
]
```

Blake2b applies **12 rounds** but SIGMA only has **10 rows**.
Rounds 10 and 11 reuse rows 0 and 1 (`r % 10`).

```python
# In _compress():
s = SIGMA[r % 10]   # r goes from 0 to 11, r % 10 gives 0,1,2,...,9,0,1
```

> **Why permute the order?**
> If words were always read in the same order, some parts of the message would be less mixed than others. SIGMA ensures that every word of the message influences all bits of the hash, regardless of its position.

### 4.3 Rotation constants

```python
# In crypto/blake2b.py
R1 = 32   # rotation in step 2 of G
R2 = 24   # rotation in step 4 of G
R3 = 16   # rotation in step 6 of G
R4 = 63   # rotation in step 8 of G
```

These 4 values **define Blake2b** and distinguish it from Blake2s (which uses 16, 12, 8, 7).
They were chosen to maximize bit mixing on 64-bit words.

---

## 5. Data representation: bytes and 64-bit words

Blake2b works internally on **64-bit integers**.
All data (message blocks, state, output) must be converted between bytes and 64-bit integers.

### 5.1 Bytes -> 64-bit words (little-endian)

"Little-endian" = the **least significant** byte comes **first**.

```
Bytes : [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
                                                            ^-- most significant byte
          ^-- least significant byte

64-bit word = 0x0807060504030201
```

In code:

```python
# In crypto/blake2b.py : _bytes_to_words_16()
def _bytes_to_words_16(block: bytes) -> list:
    words = []
    for i in range(16):              # 16 words in a 128-byte block
        word = 0
        for byte_pos in range(8):    # 8 bytes per 64-bit word
            octet = block[i * 8 + byte_pos]
            word  = word | (octet << (byte_pos * 8))
            #                         ^^^^^^^^^^^^^^^^^
            #                         shift byte to its position in the word
        words.append(word)
    return words
```

Step by step for 1 word of 8 bytes:

```
byte[0] = 0x01  ->  0x01 << 0  =  0x0000000000000001
byte[1] = 0x02  ->  0x02 << 8  =  0x0000000000000200
byte[2] = 0x03  ->  0x03 << 16 =  0x0000000000030000
...
byte[7] = 0x08  ->  0x08 << 56 =  0x0800000000000000
                                  -----------------------
                      OR of all =  0x0807060504030201
```

### 5.2 64-bit words -> bytes (little-endian)

The reverse operation: extract bytes from the word one at a time.

```python
# In crypto/blake2b.py : _state_to_bytes()
def _state_to_bytes(h: list) -> bytes:
    result = bytearray()
    for word in h:                   # 8 64-bit words
        for byte_pos in range(8):    # 8 bytes per word
            octet = (word >> (byte_pos * 8)) & 0xFF
            #         ^^^^^^^^^^^^^^^^^^^^^^^^^
            #         shift word to bring the target byte to position 0
            #                                    & 0xFF keeps only the 8 low-order bits
            result.append(octet)
    return bytes(result)
```

---

## 6. Primitive 1: bit rotation (rotr64)

Rotation is the only "special" mathematical operation in Blake2b.

### What it is

A **right rotation by n bits** on 64 bits means:
- Bits shift to the right by `n` positions
- Bits that "fall off" the right "come back in" from the left

> Unlike a **shift** (`>>`), which discards the bits that fall off, a **rotation** loses no bits. It is a **reversible** operation (left rotation = inverse right rotation).

### Visual example on 8 bits (for readability)

```
Original value : 1011 0001   (= 0xB1)
Right rotation by 3 bits:
  bits falling off the right : 001
  remaining bits              : 1011 0
                                       ^-- the 3 fallen bits come back in from the left
  Result                    : 001 1011 0  => 0010 1101  (= 0x16 on 5 bits... simplified example)
```

On 64 bits, the formula is:

```
rotr64(x, n) = (x >> n) | (x << (64 - n))   mod 2^64
                  ^          ^
                  |          |
                  |          bits wrapping back to the left
                  bits shifted to the right
```

### In code

```python
# In crypto/blake2b.py
U64 = 0xFFFFFFFFFFFFFFFF   # 64-bit mask

def _rotr64(x: int, n: int) -> int:
    right_part = x >> n              # bits shifted to the right
    left_part  = x << (64 - n)      # bits wrapping back to the left
    return (right_part | left_part) & U64
    #                                  ^^^^^
    #                                  mask to 64 bits because Python
    #                                  works with arbitrary precision integers
```

### Why use rotations?

Rotations ensure that **all input bits influence all output bits** after a few applications. This is called **diffusion**.

With a plain shift (`>>`), the high-order bits would be progressively lost. Rotation preserves and mixes them.

---

## 7. Primitive 2: the G function (the heart of Blake2b)

G is the elementary building block of the entire algorithm. If you understand G, you understand Blake2b.

### What G does

G takes **6 inputs** (4 state words + 2 message words) and returns **4 outputs** (the mixed state words).

```
Inputs : a, b, c, d  (64-bit state words)
         x, y         (words from the message block)

Outputs : a', b', c', d'  (mixed words)
```

### The 8 steps of G

```python
# In crypto/blake2b.py
def _G(a: int, b: int, c: int, d: int, x: int, y: int):

    # Step 1: mix a, b and the 1st message word
    a = (a + b + x) & U64

    # Step 2: d reacts to a via XOR then 32-bit rotation
    d = _rotr64(d ^ a, R1)     # R1 = 32

    # Step 3: c absorbs d
    c = (c + d) & U64

    # Step 4: b reacts to c via XOR then 24-bit rotation
    b = _rotr64(b ^ c, R2)     # R2 = 24

    # Step 5: mix a, b and the 2nd message word
    a = (a + b + y) & U64

    # Step 6: d reacts to a again, 16-bit rotation
    d = _rotr64(d ^ a, R3)     # R3 = 16

    # Step 7: c absorbs d again
    c = (c + d) & U64

    # Step 8: b reacts to c again, 63-bit rotation
    b = _rotr64(b ^ c, R4)     # R4 = 63

    return a, b, c, d
```

### Visualizing the data flow

```
        a          b          c          d
        |          |          |          |
     [+b+x] ------.          |          |
        |          |          |          |
        .----------.----------.--------[^ XOR]
        |          |          |        [rotr32]
        |          |          |          d'
        |          |        [+ d'] ------.
        |          |          |          |
        |        [^ XOR] ----.           |
        |        [rotr24]    |           |
        |          b'        c'          |
     [+b'+y]-------.         |           |
        |          |         |           |
        .----------.---------.---------[^ XOR]
        |          |         |          [rotr16]
        a'         |         |           d''
        |          |       [+d''] -------.
        |        [^ XOR] ---.            |
        |        [rotr63]   |            |
        a'         b''      c''         d''
```

> **Addition / XOR / rotation pattern**: addition mixes bits with carry propagation, XOR mixes without carry, rotation shifts bits without losing any. Alternating all three creates maximum diffusion.

### Step-by-step trace example

```python
# Example values (truncated for readability)
a = 0x6A09E667F3BCC908   # IV[0]
b = 0xBB67AE8584CAA73B   # IV[1]
c = 0x3C6EF372FE94F82B   # IV[2]
d = 0xA54FF53A5F1D36F1   # IV[3]
x = 0x0000000000000000   # first message word
y = 0x0000000000000000   # second message word

# Step 1
a = (0x6A09E667F3BCC908 + 0xBB67AE8584CAA73B + 0) & U64
  = 0x25718EED78876643   # overflow truncated to 64 bits

# Step 2
d = rotr64(0xA54FF53A5F1D36F1 ^ 0x25718EED78876643, 32)
  = rotr64(0x803E1BD727942292, 32)
  = 0x27942292803E1BD7   # the 32 right bits move to the left
...
```

---

## 8. Compression F: 12 rounds of G

The compression function `_compress()` applies G over an entire block.

### Structure of the working vector v

During compression, we work on a vector `v` of **16 64-bit words**.
It can be visualized as a 4x4 grid:

```
  v[ 0]   v[ 1]   v[ 2]   v[ 3]
  v[ 4]   v[ 5]   v[ 6]   v[ 7]
  v[ 8]   v[ 9]   v[10]   v[11]
  v[12]   v[13]   v[14]   v[15]
```

At the start of each compression, `v` is built as follows:
- **First half** `v[0..7]` = copy of the current state `h`
- **Second half** `v[8..15]` = `IV` constants

```python
# In crypto/blake2b.py : _compress()
v = [0] * 16
for i in range(8):
    v[i] = h[i]        # first half: current state
for i in range(8):
    v[i + 8] = IV[i]   # second half: IV constants
```

### Counter injection and finalization flag

Before starting the rounds, `v[12]` and `v[13]` are modified to inject the **byte counter** (how many bytes have been processed in total up to this block):

```python
t_low  = t & U64           # counter low 64 bits
t_high = (t >> 64) & U64   # counter high 64 bits (= 0 in practice)
v[12] = v[12] ^ t_low
v[13] = v[13] ^ t_high
```

> **Why a counter?** So that two identical messages of different lengths (e.g. `"abc"` and `"abc\x00\x00"`) produce different hashes. The counter encodes the true length of the original message.

If this is the **last block**, all bits of `v[14]` are inverted:

```python
if last:
    v[14] = v[14] ^ U64   # U64 = 0xFFFF...FFFF  => NOT on 64 bits
```

This flag signals to the algorithm that the compression must "finalize" the hash.

### The 12 rounds of G (columns then diagonals)

Each round applies G **8 times**: 4 on the columns of the square, 4 on the diagonals.

```
Round r:
  Select s = SIGMA[r % 10]     <- message permutation for this round

  G on column 0 : (v[0], v[4], v[ 8], v[12]) with m[s[0]], m[s[1]]
  G on column 1 : (v[1], v[5], v[ 9], v[13]) with m[s[2]], m[s[3]]
  G on column 2 : (v[2], v[6], v[10], v[14]) with m[s[4]], m[s[5]]
  G on column 3 : (v[3], v[7], v[11], v[15]) with m[s[6]], m[s[7]]

  G on diag.  0 : (v[0], v[5], v[10], v[15]) with m[s[8]],  m[s[9]]
  G on diag.  1 : (v[1], v[6], v[11], v[12]) with m[s[10]], m[s[11]]
  G on diag.  2 : (v[2], v[7], v[ 8], v[13]) with m[s[12]], m[s[13]]
  G on diag.  3 : (v[3], v[4], v[ 9], v[14]) with m[s[14]], m[s[15]]
```

Visualization of the 4x4 square: columns (|) then diagonals (/)

```
Columns:           Diagonals:
|  |  |  |         /  /  /  /
0  1  2  3        0  1  2  3
4  5  6  7        5  6  7  4
8  9 10 11       10 11  8  9
12 13 14 15      15 12 13 14
```

> Columns mix elements "vertically". Diagonals mix elements "diagonally". After one column pass AND one diagonal pass, every element has interacted with all other elements of the square.

In code:

```python
# In crypto/blake2b.py : _compress()
for r in range(12):
    s = SIGMA[r % 10]

    # 4 G on columns
    v[0], v[4], v[8],  v[12] = _G(v[0], v[4], v[8],  v[12], m[s[0]],  m[s[1]])
    v[1], v[5], v[9],  v[13] = _G(v[1], v[5], v[9],  v[13], m[s[2]],  m[s[3]])
    v[2], v[6], v[10], v[14] = _G(v[2], v[6], v[10], v[14], m[s[4]],  m[s[5]])
    v[3], v[7], v[11], v[15] = _G(v[3], v[7], v[11], v[15], m[s[6]],  m[s[7]])

    # 4 G on diagonals
    v[0], v[5], v[10], v[15] = _G(v[0], v[5], v[10], v[15], m[s[8]],  m[s[9]])
    v[1], v[6], v[11], v[12] = _G(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]])
    v[2], v[7], v[8],  v[13] = _G(v[2], v[7], v[8],  v[13], m[s[12]], m[s[13]])
    v[3], v[4], v[9],  v[14] = _G(v[3], v[4], v[9],  v[14], m[s[14]], m[s[15]])
```

### Updating state h

After 12 rounds, state `h` is updated:

```python
for i in range(8):
    h[i] = h[i] ^ v[i] ^ v[i + 8]
```

Formula: `h[i] = h_old[i]  XOR  v[i]  XOR  v[i+8]`

Breaking it down:
- `v[i]` = transformed version of `h[i]` after 12 rounds
- `v[i+8]` = transformed version of `IV[i]` after 12 rounds
- `h[i] XOR v[i]` = XOR between old and new state
- XOR with `v[i+8]` incorporates the transformed IV to strengthen the hash

> **Why XOR with the old `h[i]`?** This is the Davies-Meyer construction. It ensures that if the message block is entirely zero, the hash does not fall back to the original IV values. It breaks a dangerous symmetry.

---

## 9. Main algorithm: blake2b()

Now that we understand the building blocks, here is the complete algorithm.

### Signature

```python
def blake2b(data: bytes, digest_size: int = 64, key: bytes = b'') -> bytes:
```

| Parameter | Type | Description |
|---|---|---|
| `data` | `bytes` | Data to hash (any size) |
| `digest_size` | `int` | Desired hash length in bytes, from 1 to 64 |
| `key` | `bytes` | Optional key for MAC mode (0 to 64 bytes) |

### Step 1: initialize h

```python
h = [0] * 8
for i in range(8):
    h[i] = IV[i]   # copy the 8 IV constants
```

### Step 2: configure h[0] with parameters

The first word `h[0]` is modified to encode the hash parameters.
This is what makes `blake2b(data, 32)` and `blake2b(data, 64)` produce different results.

```python
key_len   = len(key)
parameter = 0x01010000 ^ (key_len << 8) ^ digest_size
h[0] = h[0] ^ parameter
```

The `parameter` word encodes 4 fields of 8 bits each:

```
Bits 24-31 : depth  = 0x01  (tree depth = 1 in sequential mode)
Bits 16-23 : fanout = 0x01  (tree width = 1 in sequential mode)
Bits  8-15 : key_len         (key length)
Bits  0-7  : digest_size     (desired hash length)

0x01010000 encodes depth=1 and fanout=1 in the correct positions.
key_len and digest_size are added via XOR.
```

Example: `blake2b(data, digest_size=32, key=b"")`:
```
parameter = 0x01010000 ^ (0 << 8) ^ 32
          = 0x01010000 ^ 0x00000000 ^ 0x00000020
          = 0x01010020
```

### Step 3: process data block by block

```python
# In crypto/blake2b.py : blake2b()

if len(data) == 0:
    # Special case: empty message -> compress a block of zeros
    _compress(h, b'\x00' * 128, t=0, last=True)
else:
    data_len = len(data)
    offset   = 0

    # All blocks except the last
    while offset + 128 < data_len:
        block   = data[offset : offset + 128]
        counter = offset + 128          # bytes processed so far
        _compress(h, block, t=counter, last=False)
        offset += 128

    # Last block (may be < 128 bytes)
    last_block   = data[offset:]
    last_padded  = last_block + b'\x00' * (128 - len(last_block))
    _compress(h, last_padded, t=data_len, last=True)
```

> **The counter `t`** passed to `_compress()` encodes the number of bytes **actually** processed. For the last padded block, we pass `data_len` (the real size), not 128.

### Step 4: serialize the hash

```python
raw = _state_to_bytes(h)   # 8 words x 8 bytes = 64 bytes
return raw[:digest_size]   # truncate to desired length
```

### Complete traced example

```python
# blake2b(b"abc", digest_size=32)

# Step 1
h = [0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, ..., 0x5BE0CD19137E2179]

# Step 2: parameter = 0x01010020  (digest_size=32, key_len=0)
h[0] = h[0] ^ 0x01010020 = 0x6A09E667F3BCC908 ^ 0x01010020 = 0x6B08C667F3BCC928

# Step 3: "abc" = 3 bytes, 1 single block (< 128 bytes)
# Block = b"abc" + b"\x00" * 125  (padded to 128 bytes)
# t = 3  (3 bytes processed)
# last = True
_compress(h, b"abc\x00...\x00", t=3, last=True)

# Step 4: serialize the 8 words into 64 bytes, keep the first 32
hash = _state_to_bytes(h)[:32]
# => bddd813c634239723171ef3fee98579b94964e3bb1cb3e427262c8c068d52319
```

---

## 10. Keyed mode (MAC)

Blake2b can operate as a **MAC (Message Authentication Code)**: with a secret key, only someone who knows the key can compute the same hash.

### How it works

The key is "prepended" to the message as a 128-byte block (key padded with zeros).

```python
# In crypto/blake2b.py : blake2b()
if key_len > 0:
    key_block = key + b'\x00' * (128 - key_len)   # key padded to 128 bytes
    data = key_block + data                        # prepend to message
```

The key length is also encoded in the parameter (see Step 2).

```
Data with key:

[  padded key (128 bytes)  ][  original data ...  ]
        Block 1                    Following blocks
```

### Example

```python
from crypto.blake2b import blake2b

# Without key (plain hash)
h1 = blake2b(b"message")

# With key (MAC)
h2 = blake2b(b"message", key=b"secret")

# h1 != h2 even if the message is identical
# Without the key "secret", it is impossible to recompute h2
```

---

## 11. Why it is secure: intuition without maths

### The avalanche effect

Changing **1 bit** in the input changes approximately **50% of the bits** in the hash.

Why? The G function mixes 4 words at a time. After 1 call to G, the change propagates to 4 words. After 8 calls (1 full round), it has propagated to all 16 positions of `v`. After 12 rounds, every bit of the hash depends on every bit of the input.

This is **diffusion**: the mixing cascades outward.

### Non-reversibility

Why can't Blake2b be "inverted"?

Because addition (`+`) is not invertible given only XOR and rotation as additional information. The combination `a = (a + b + x) & U64` followed by `d = rotr(d ^ a, 32)` creates non-linear dependencies: knowing `a'` and `d'` is not enough to recover `a`, `b`, `d` and `x` simultaneously (too many unknowns).

### Collision resistance

Finding two different messages with the same hash requires $2^{128}$ operations (for digest_size=32). The age of the universe is ~$4 \times 10^{17}$ seconds. Even at $10^{18}$ hashes/second, it would take $10^{20}$ times the age of the universe.

---

## 12. Blake2b vs Blake2s vs SHA-512

| | Blake2b | Blake2s | SHA-512 |
|---|---|---|---|
| Word size | 64 bits | 32 bits | 64 bits |
| Block size | 128 bytes | 64 bytes | 128 bytes |
| Rounds | 12 | 10 | 80 |
| Max hash | 64 bytes | 32 bytes | 64 bytes |
| Rotations (G) | 32, 24, 16, 63 | 16, 12, 8, 7 | N/A |
| Natively keyed | Yes | Yes | No (external HMAC) |
| Speed (64-bit PC) | Very fast | Slower | Slower |
| IV | Same as SHA-512 | Derived from sqrt of primes | Same |

**Blake2b is optimized for 64-bit processors** (PCs, servers). On a 32-bit processor (IoT, microcontrollers), Blake2s would be preferable.

---

## 13. Test vectors

These values allow verifying an implementation. They come from RFC 7693 and `hashlib.blake2b` (Python reference).

```python
# Format: blake2b(data, digest_size=32).hex()

blake2b(b"",    32) == "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8"
blake2b(b"abc", 32) == "bddd813c634239723171ef3fee98579b94964e3bb1cb3e427262c8c068d52319"
blake2b(b"The quick brown fox jumps over the lazy dog", 32) == "01718cec35cd3d796dd00020e0bfecb473ad23457d063b75eff29c0ffa2e58a9"

# Format: blake2b(data, digest_size=64).hex()

blake2b(b"",    64) == "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce"
blake2b(b"abc", 64) == "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"

# Keyed mode
blake2b(b"abc", 32, key=b"key") != blake2b(b"abc", 32)   # always different
```

To run the full test suite (54 vectors):

```bash
python crypto/blake2b_test.py
# Expected: 54/54 PASS
```

---

## 14. Complete summary diagram

```
BLAKE2B - COMPLETE DIAGRAM
==========================

INPUTS
------
  data        : message of any size
  digest_size : 1..64 bytes (desired hash length)
  key         : 0..64 bytes (optional, MAC mode)


STEP 0 - PREPARATION
---------------------
  If key provided:
    data = [key + zeros(128 - len(key))] + data
                                          ^-- prepend

  Encode parameters into a 64-bit word:
    parameter = 0x01010000 XOR (key_len << 8) XOR digest_size


STEP 1 - INITIALIZE H
----------------------
  h[0..7] = IV[0..7]
  h[0]    = h[0] XOR parameter


STEP 2 - LOOP OVER BLOCKS (128 bytes each)
-------------------------------------------
  For each block[i] of 128 bytes:
    t    = cumulative_bytes_processed    (counter)
    last = True if last block

    COMPRESSION F(h, block, t, last):

      [1] Decode block -> m[0..15]  (16 64-bit words, little-endian)

      [2] Build v[0..15]:
            v[0..7]  = h[0..7]
            v[8..15] = IV[0..7]

      [3] Inject counter:
            v[12] = v[12] XOR t

      [4] If last:
            v[14] = v[14] XOR 0xFFFFFFFFFFFFFFFF

      [5] 12 rounds:
            For r = 0..11:
              s = SIGMA[r % 10]

              COLUMNS (4 x G):
                G(v[0], v[4], v[ 8], v[12], m[s[0]],  m[s[1]])
                G(v[1], v[5], v[ 9], v[13], m[s[2]],  m[s[3]])
                G(v[2], v[6], v[10], v[14], m[s[4]],  m[s[5]])
                G(v[3], v[7], v[11], v[15], m[s[6]],  m[s[7]])

              DIAGONALS (4 x G):
                G(v[0], v[5], v[10], v[15], m[s[8]],  m[s[9]])
                G(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]])
                G(v[2], v[7], v[ 8], v[13], m[s[12]], m[s[13]])
                G(v[3], v[4], v[ 9], v[14], m[s[14]], m[s[15]])

      [6] Update h:
            For i = 0..7:
              h[i] = h[i] XOR v[i] XOR v[i+8]


THE G FUNCTION
--------------
  G(a, b, c, d, x, y):
    a = (a + b + x) mod 2^64
    d = rotr64(d XOR a, 32)
    c = (c + d)     mod 2^64
    b = rotr64(b XOR c, 24)
    a = (a + b + y) mod 2^64
    d = rotr64(d XOR a, 16)
    c = (c + d)     mod 2^64
    b = rotr64(b XOR c, 63)
    return a, b, c, d


ROTATION
--------
  rotr64(x, n) = ((x >> n) | (x << (64 - n))) AND 0xFFFFFFFFFFFFFFFF


STEP 3 - OUTPUT
---------------
  raw  = h[0..7] serialized in little-endian (64 bytes)
  hash = raw[0 : digest_size]


CONSTANTS
---------
  IV = [0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B,
        0xA54FF53A5F1D36F1, 0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
        0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179]
        (fractional parts of the square roots of 2,3,5,7,11,13,17,19)

  SIGMA = 10 rows of 16 indices, defined in the code
  Rotations: R1=32, R2=24, R3=16, R4=63


OPERATION COUNT PER HASH (128-byte message)
--------------------------------------------
  1 compression
    x  12 rounds
       x  8 calls to G
          x  8 operations (4 additions + 2 XOR + 2 rotations)
  = 768 elementary operations
  + 8 XOR for the h update
  = 776 elementary operations for a 128-byte block
```

