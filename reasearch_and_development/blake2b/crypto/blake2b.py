#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blake2b - Implementation from scratch
======================================
Reference: RFC 7693  (https://www.rfc-editor.org/rfc/rfc7693)

NO OPTIMIZATION - raw pedagogical code
----------------------------------------
Each operation is written explicitly, step by step.
Priority: readability and faithfulness to the mathematical specification.

GENERAL PRINCIPLE
-----------------
Blake2b is a cryptographic hash function.
It transforms any amount of data into a fixed-size digest
(from 1 to 64 bytes), through a series of non-reversible
mathematical transformations.

INTERNAL STRUCTURE
------------------

    [input data]
           |
     split into 128-byte blocks
           |
     +-----------+     +-----------+     +-----------+
     |  block 1  | --> |  block 2  | --> |   last    |
     | compress  |     | compress  |     | compress  |
     +-----------+     +-----------+     +-----------+
           |
     state h (8 words of 64 bits)
           |
     truncate to digest_size bytes
           |
     [final hash]

THE G FUNCTION (core of the algorithm)
---------------------------------------
Blake2b mixes data via the G function which operates on
4 words of 64 bits (a, b, c, d) and two message words (x, y):

    Step 1: a = (a + b + x) mod 2^64
    Step 2: d = right_rotate(d XOR a, 32)
    Step 3: c = (c + d) mod 2^64
    Step 4: b = right_rotate(b XOR c, 24)
    Step 5: a = (a + b + y) mod 2^64
    Step 6: d = right_rotate(d XOR a, 16)
    Step 7: c = (c + d) mod 2^64
    Step 8: b = right_rotate(b XOR c, 63)

G is called 8 times per round: 4 times on the columns,
4 times on the diagonals of the vector v arranged as 4x4.
Blake2b applies 12 rounds per block.

USAGE IN RANDOMX
-----------------
    hash256(data) = blake2b(data, digest_size=32)  -> seed / final result
    hash512(data) = blake2b(data, digest_size=64)  -> large internal seeds
"""

import sys
import io

# -- UTF-8 encoding on Windows console ------------------------------------
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============================================================
# CONSTANTS
# ============================================================

# 64-bit mask for modular arithmetic (mod 2^64)
U64 = 0xFFFFFFFFFFFFFFFF

# Initialization vector (IV) - same values as SHA-512
# Derived from the fractional parts of the square roots
# of the first 8 prime numbers (2, 3, 5, 7, 11, 13, 17, 19)
IV = [
    0x6A09E667F3BCC908,   # sqrt of 2
    0xBB67AE8584CAA73B,   # sqrt of 3
    0x3C6EF372FE94F82B,   # sqrt of 5
    0xA54FF53A5F1D36F1,   # sqrt of 7
    0x510E527FADE682D1,   # sqrt of 11
    0x9B05688C2B3E6C1F,   # sqrt of 13
    0x1F83D9ABFB41BD6B,   # sqrt of 17
    0x5BE0CD19137E2179,   # sqrt of 19
]

# Sigma table: message permutation for each round.
# 10 rows (rounds 0 to 9).
# Blake2b has 12 rounds: rounds 10 and 11 reuse sigma[0] and sigma[1].
# Each row contains 16 indices indicating the reading order of
# message words for the 8 G calls of that round.
SIGMA = [
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    [14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3],
    [11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4],
    [ 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8],
    [ 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13],
    [ 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9],
    [12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11],
    [13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10],
    [ 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5],
    [10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0],
]

# Rotation constants used by the G function.
# These 4 values define the "signature" of Blake2b
# (different from Blake2s which uses R1=16, R2=12, R3=8, R4=7)
R1 = 32
R2 = 24
R3 = 16
R4 = 63


# ============================================================
# STEP 0 - Block encoding
#
# A Blake2b block = 128 bytes = 16 words of 64 bits in little-endian.
# The final serialization = 8 words of 64 bits in little-endian.
# We work with Python integers for readability.
# ============================================================

def _bytes_to_words_16(block: bytes) -> list:
    """
    Converts a 128-byte block into a list of 16 64-bit integers (little-endian).

    Raw procedure: read 8 bytes at a time, build each word manually.
        word = byte[0]
             + byte[1] * 256
             + byte[2] * 256^2
             + ...
             + byte[7] * 256^7
    """
    words = []
    for i in range(16):              # 16 words in a 128-byte block
        word = 0
        for byte_pos in range(8):    # 8 bytes per 64-bit word
            byte_val = block[i * 8 + byte_pos]
            word     = word | (byte_val << (byte_pos * 8))
        words.append(word)
    return words


def _state_to_bytes(h: list) -> bytes:
    """
    Converts the final state h (8 64-bit integers) to 64 bytes (little-endian).

    Raw procedure: write each word byte by byte.
        byte[k] = (word >> (k * 8)) & 0xFF
    """
    result = bytearray()
    for word in h:                   # 8 64-bit words
        for byte_pos in range(8):    # 8 bytes per word
            byte_val = (word >> (byte_pos * 8)) & 0xFF
            result.append(byte_val)
    return bytes(result)


# ============================================================
# STEP 1 - Mathematical primitives
# ============================================================

def _rotr64(x: int, n: int) -> int:
    """
    Circular right rotation on 64 bits.

    Bits that "fall off" the right "wrap around" to the left.
    This is a REVERSIBLE operation: no bit is lost.

    Example on 8 bits: rotr(0b10110001, 3) = 0b00110110
    Formula: (x >> n) | (x << (64 - n))  mod 2^64
    """
    right_part = x >> n              # bits shifted to the right
    left_part  = x << (64 - n)      # bits that "wrap around" to the left
    return (right_part | left_part) & U64


# ============================================================
# STEP 2 - Mixing function G (RFC 7693, Section 3.1)
#
# G is the core of Blake2b.
# It takes 4 words (a, b, c, d) and 2 message words (x, y).
# It mixes them in 8 steps and returns the 4 updated words.
#
# The addition / XOR / rotation scheme maximizes diffusion:
# after G, each output bit depends on all input bits.
# ============================================================

def _G(a: int, b: int, c: int, d: int, x: int, y: int):
    """
    Blake2b mixing function G (RFC 7693, Section 3.1).

    Takes 4 64-bit words and 2 message words.
    Returns the 4 mixed words (a, b, c, d).

    All additions are modulo 2^64 (natural uint64 behavior in C).

    Step 1: a = (a + b + x) mod 2^64    <- addition + injection of 1st message word
    Step 2: d = rotr64(d XOR a, 32)     <- 32-bit rotation
    Step 3: c = (c + d) mod 2^64        <- addition
    Step 4: b = rotr64(b XOR c, 24)     <- 24-bit rotation
    Step 5: a = (a + b + y) mod 2^64    <- addition + injection of 2nd message word
    Step 6: d = rotr64(d XOR a, 16)     <- 16-bit rotation
    Step 7: c = (c + d) mod 2^64        <- addition
    Step 8: b = rotr64(b XOR c, 63)     <- 63-bit rotation
    """
    # Step 1
    a = (a + b + x) & U64
    # Step 2
    d = _rotr64(d ^ a, R1)
    # Step 3
    c = (c + d) & U64
    # Step 4
    b = _rotr64(b ^ c, R2)
    # Step 5
    a = (a + b + y) & U64
    # Step 6
    d = _rotr64(d ^ a, R3)
    # Step 7
    c = (c + d) & U64
    # Step 8
    b = _rotr64(b ^ c, R4)

    return a, b, c, d


# ============================================================
# STEP 3 - Compression function F (RFC 7693, Section 3.2)
#
# Takes the current state h (8 words) and a 128-byte block.
# Mixes the block into h via 12 rounds of the G function.
# Modifies h in place.
#
# The working vector v (16 words) is visualized as a 4x4 square:
#
#   v[ 0]  v[ 1]  v[ 2]  v[ 3]
#   v[ 4]  v[ 5]  v[ 6]  v[ 7]
#   v[ 8]  v[ 9]  v[10]  v[11]
#   v[12]  v[13]  v[14]  v[15]
#
# Columns   : (0,4,8,12) (1,5,9,13) (2,6,10,14) (3,7,11,15)
# Diagonals : (0,5,10,15) (1,6,11,12) (2,7,8,13) (3,4,9,14)
# ============================================================

def _compress(h: list, block: bytes, t: int, last: bool) -> None:
    """
    Compression function F (RFC 7693, Section 3.2).

    Parameters
    ----------
    h     : internal state, list of 8 uint64 integers (modified in place)
    block : exactly 128 bytes (zero-padded if necessary)
    t     : total number of bytes processed up to the end of this block
            (128-bit counter, but t < 2^64 in practice)
    last  : True if this is the last block (activates the finalization flag)

    Step-by-step procedure
    ----------------------
    1. Decode the block into 16 little-endian uint64 words -> m[0..15]
    2. Build the working vector v[0..15]:
           v[0..7]  = current state h
           v[8..15] = initialization vector IV
    3. Inject the counter t into v[12] and v[13] (XOR)
    4. If last block: invert all bits of v[14] (end flag)
    5. Execute 12 mixing rounds (columns then diagonals)
    6. Update h: h[i] ^= v[i] ^ v[i+8]  for i = 0..7
    """

    # --- Step 1: decode the 128 bytes of the block into 16 uint64 words ---
    # Raw procedure: read 8 bytes at a time, word by word
    m = _bytes_to_words_16(block)

    # --- Step 2: build the working vector v (16 words) ---
    # v[0..7]  = copy of the current state h
    # v[8..15] = IV constants
    # Built explicitly word by word.
    v = [0] * 16
    for i in range(8):
        v[i] = h[i]        # first half: current state
    for i in range(8):
        v[i + 8] = IV[i]   # second half: IV constants

    # --- Step 3: inject the byte counter ---
    # t is a 128-bit counter encoded as two uint64 (t_low, t_high).
    # In practice t < 2^64 so t_high = 0.
    t_low  = t & U64             # low 64 bits
    t_high = (t >> 64) & U64    # high 64 bits (always 0 here)
    v[12] = v[12] ^ t_low
    v[13] = v[13] ^ t_high

    # --- Step 4: finalization flag ---
    # On the last block, invert all bits of v[14].
    # XOR with U64 (= 0xFFFFFFFFFFFFFFFF) is equivalent to NOT on 64 bits.
    if last:
        v[14] = v[14] ^ U64

    # --- Step 5: 12 mixing rounds ---
    # Each round applies G 8 times:
    #   4 times on the columns of the 4x4 square
    #   4 times on the diagonals of the 4x4 square
    for r in range(12):
        # Select the SIGMA permutation for this round.
        # SIGMA has 10 rows, rounds 10 and 11 cycle (10%10=0, 11%10=1).
        s = SIGMA[r % 10]

        # -- 4 G calls on columns --
        # Column 0: indices (0, 4,  8, 12)
        v[0], v[4], v[8],  v[12] = _G(v[0], v[4], v[8],  v[12], m[s[0]],  m[s[1]])
        # Column 1: indices (1, 5,  9, 13)
        v[1], v[5], v[9],  v[13] = _G(v[1], v[5], v[9],  v[13], m[s[2]],  m[s[3]])
        # Column 2: indices (2, 6, 10, 14)
        v[2], v[6], v[10], v[14] = _G(v[2], v[6], v[10], v[14], m[s[4]],  m[s[5]])
        # Column 3: indices (3, 7, 11, 15)
        v[3], v[7], v[11], v[15] = _G(v[3], v[7], v[11], v[15], m[s[6]],  m[s[7]])

        # -- 4 G calls on diagonals --
        # Diagonal 0: indices (0,  5, 10, 15)
        v[0], v[5], v[10], v[15] = _G(v[0], v[5], v[10], v[15], m[s[8]],  m[s[9]])
        # Diagonal 1: indices (1,  6, 11, 12)
        v[1], v[6], v[11], v[12] = _G(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]])
        # Diagonal 2: indices (2,  7,  8, 13)
        v[2], v[7], v[8],  v[13] = _G(v[2], v[7], v[8],  v[13], m[s[12]], m[s[13]])
        # Diagonal 3: indices (3,  4,  9, 14)
        v[3], v[4], v[9],  v[14] = _G(v[3], v[4], v[9],  v[14], m[s[14]], m[s[15]])

    # --- Step 6: update state h ---
    # Formula: h[i] = h[i] XOR v[i] XOR v[i+8]
    # XOR with v[i+8] incorporates the transformed IV into the final state,
    # ensuring that even a state h = 0 would produce a non-zero hash.
    for i in range(8):
        h[i] = h[i] ^ v[i] ^ v[i + 8]


# ============================================================
# MAIN ALGORITHM
# ============================================================

def blake2b(data: bytes, digest_size: int = 64, key: bytes = b'') -> bytes:
    """
    Computes the Blake2b hash of the input data (RFC 7693).

    Parameters
    ----------
    data        : data to hash (bytes, arbitrary size)
    digest_size : hash size in bytes  [1 .. 64]
                  32 = 256 bits  |  64 = 512 bits
    key         : optional key (0 to 64 bytes)
                  Allows using Blake2b as a MAC.

    Returns
    -------
    bytes of length digest_size

    Step-by-step procedure (RFC 7693)
    -----------------------------------
    1. Initialize h[0..7] = IV[0..7]  (copy of the constants)
    2. Apply the "parameter block": XOR h[0] with the parameters
       (digest_size, key length, fanout=1, depth=1)
    3. If key: create a 128-byte key block (key + zero padding)
       and prepend it to the data
    4. Split the data into 128-byte blocks
    5. Compress each block with _compress():
         - All blocks except the last: last=False
         - The last block: zero-padded if incomplete, last=True
    6. Serialize h in little-endian (64 bytes) and truncate to digest_size
    """
    assert 1 <= digest_size <= 64, "digest_size must be between 1 and 64 bytes"
    assert len(key) <= 64,         "key cannot exceed 64 bytes"

    # --- Step 1: initialize state h with IV constants ---
    h = [0] * 8
    for i in range(8):
        h[i] = IV[i]

    # --- Step 2: apply the "parameter block" ---
    # The first word of the parameter block encodes the main parameters:
    #
    #   Bits  0-7  : digest_size  (desired hash length)
    #   Bits  8-15 : key_len      (key length)
    #   Bits 16-23 : fanout = 1   (hash tree, always 1 in sequential mode)
    #   Bits 24-31 : depth  = 1   (tree depth, always 1 in sequential mode)
    #
    # 0x01010000 encodes fanout=1 and depth=1 in the correct bits.
    # XOR this word with h[0] to "configure" the initial state.
    key_len   = len(key)
    parameter = 0x01010000 ^ (key_len << 8) ^ digest_size
    h[0]      = h[0] ^ parameter

    # --- Step 3: prepare the data (key injection if present) ---
    # If a key is provided, it becomes the first 128-byte block.
    # The key is zero-padded to 128 bytes.
    if key_len > 0:
        key_block = key + b'\x00' * (128 - key_len)   # key padded to 128 bytes
        data      = key_block + data                   # prepend to message

    # --- Steps 4 and 5: split and compress block by block ---

    if len(data) == 0:
        # Special case: empty message (or empty key with no data).
        # We must still compress a zero block to produce
        # a valid hash (the specification requires this explicitly).
        empty_block = b'\x00' * 128
        counter     = 0       # 0 message bytes processed
        _compress(h, empty_block, t=counter, last=True)

    else:
        data_len = len(data)
        offset   = 0

        # Process all complete blocks EXCEPT the last one.
        # We stop at (data_len - 128) to guarantee that at least
        # one byte always remains for the "last block".
        while offset + 128 < data_len:
            block   = data[offset : offset + 128]
            counter = offset + 128   # bytes processed at the end of this block
            _compress(h, block, t=counter, last=False)
            offset  = offset + 128

        # Process the last block (may be incomplete, < 128 bytes).
        # Zero-padded to exactly 128 bytes.
        last_block    = data[offset:]
        pad_length    = 128 - len(last_block)
        last_padded   = last_block + b'\x00' * pad_length
        counter       = data_len   # counter = total real message length
        _compress(h, last_padded, t=counter, last=True)

    # --- Step 6: serialize the final state and truncate ---
    # The state h (8 64-bit words) is serialized in little-endian.
    # Raw procedure: write each word byte by byte.
    # Then truncate to digest_size bytes.
    raw = _state_to_bytes(h)     # 64 bytes (8 words x 8 bytes)
    return raw[:digest_size]     # truncate to the desired size


# ============================================================
# SHORTCUTS FOR RANDOMX
# ============================================================

def hash256(data: bytes) -> bytes:
    """
    Blake2b with 256-bit output (32 bytes).

    Usage in RandomX:
        - Final result: R = Hash256(RegisterFile)
        - AES generator keys

    Equivalent to: hashlib.blake2b(data, digest_size=32).digest()
    """
    return blake2b(data, digest_size=32)


def hash512(data: bytes) -> bytes:
    """
    Blake2b with 512-bit output (64 bytes).

    Usage in RandomX:
        - Initial seed       : S = Hash512(H)
        - gen4 re-seeding    : S = Hash512(RegisterFile)
        - BlakeGenerator init: state = Hash512(padded_key)

    Equivalent to: hashlib.blake2b(data, digest_size=64).digest()
    """
    return blake2b(data, digest_size=64)


# ============================================================
# QUICK DEMO (if run directly)
# ============================================================

if __name__ == '__main__':
    import hashlib

    print("=" * 55)
    print("  Blake2b - Implementation from scratch")
    print("=" * 55)

    tests = [
        b"",
        b"abc",
        b"Hello, RandomX!",
        b"x" * 128,   # exactly 1 block
        b"x" * 129,   # 1 block + 1 byte
        b"x" * 256,   # exactly 2 blocks
    ]

    all_ok = True
    for data in tests:
        our256 = hash256(data).hex()
        ref256 = hashlib.blake2b(data, digest_size=32).digest().hex()
        our512 = hash512(data).hex()
        ref512 = hashlib.blake2b(data, digest_size=64).digest().hex()
        ok256  = "OK  " if our256 == ref256 else "FAIL"
        ok512  = "OK  " if our512 == ref512 else "FAIL"
        label  = repr(data) if len(data) <= 20 else f"b'...' ({len(data)} bytes)"
        print(f"\n  Input : {label}")
        print(f"  256-bit [{ok256}] : {our256[:48]}...")
        print(f"  512-bit [{ok512}] : {our512[:48]}...")
        if ok256 == "FAIL" or ok512 == "FAIL":
            all_ok = False
            print(f"  EXPECTED 256 : {ref256[:48]}...")
            print(f"  EXPECTED 512 : {ref512[:48]}...")

    print("\n" + "=" * 55)
    print(f"  Overall result: {'ALL OK' if all_ok else 'FAILURE(S) DETECTED'}")
    print("=" * 55)
