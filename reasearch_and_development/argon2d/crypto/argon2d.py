#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
argon2d.py  -  Argon2 from scratch (modes d, i, id)
=====================================================
Reference: RFC 9106  (https://www.rfc-editor.org/rfc/rfc9106)

NO OPTIMIZATIONS - raw pedagogical code
-----------------------------------------
Each operation is written explicitly, step by step.
Priority: readability and fidelity to the mathematical specification.

GENERAL PRINCIPLE
-----------------
Argon2 is a memory-oriented key derivation function (KDF).
It resists GPU/ASIC attacks because it requires a large
memory space (m KiB) for each computation. The adversary cannot
massively parallelize: each core needs the same RAM.

Three modes:
  - Argon2d  (type 0): DATA-DEPENDENT memory access
                        J1/J2 come from the previous block → GPU resistance
  - Argon2i  (type 1): DATA-INDEPENDENT memory access
                        J1/J2 generated via Blake2b → side-channel resistance
  - Argon2id (type 2): HYBRID
                        pass 0 segments 0-1 = Argon2i
                        remainder = Argon2d

GENERAL STRUCTURE (step by step)
----------------------------------

  STEP 1 — Initial hash H0
  ─────────────────────────
  H0 = Blake2b-64( LE32(p) || LE32(T) || LE32(m) || LE32(t) ||
                   LE32(version) || LE32(type) ||
                   LE32(|pwd|) || pwd ||
                   LE32(|salt|) || salt ||
                   LE32(|K|)   || K   ||
                   LE32(|X|)   || X   )

  STEP 2 — Memory allocation
  ─────────────────────────
  m' blocks of 1024 bytes, organized in p lanes of q columns.
      m' = 4*p * floor(m / (4*p))   (rounded to multiple of 4*p)
      q  = m' / p                    (columns per lane)
      s  = q / 4                     (blocks per segment, SYNC_POINTS=4)

  STEP 3 — Initialize the first two blocks of each lane
  ────────────────────────────────────────────────────────
  B[i][0] = H'( H0 || LE32(0) || LE32(i),  T=1024 )
  B[i][1] = H'( H0 || LE32(1) || LE32(i),  T=1024 )
  Where H' = Blake2b variable-length hash (RFC 9106 §3.2)

  STEP 4 — Fill loop  (t passes, 4 segments, p lanes)
  ──────────────────────────────────────────────────────
  For each position (pass, segment, lane, index):
    Choose a reference block via J1 and J2 (data-dependent or pseudo-random)
    B[lane][col] = G( B[lane][prev], B[ref_lane][ref_col] )
    If pass > 0: B[lane][col] ^= old_value  (XOR mode)

  STEP 5 — Finalization
  ───────────────────────
  C = B[0][q-1] XOR B[1][q-1] XOR ... XOR B[p-1][q-1]
  tag = H'( C, T )


COMPRESSION FUNCTION G  (RFC 9106 §3.4)
────────────────────────────────────────
  G( X, Y ):
    1.  R = X XOR Y                (XOR word by word, 128 words of 64 bits)
    2.  Q = copy of R
    3.  Apply P to the 8 ROWS of Q
        (consecutive words  16*l .. 16*l+15  for l = 0..7)
    4.  Apply P to the 8 COLUMNS of Q
        (interleaved words  2*l, 2*l+1, 2*l+16, 2*l+17 ... for l = 0..7)
    5.  Z = Q XOR R
    G(X,Y) = Z

  P is the Blake2b permutation on 16 words (BLAKE2_ROUND_NOMSG):
    4 G_mix calls on the columns of the 4x4 square, then 4 on the diagonals.

  G_mix (DIFFERENT from Blake2b because it uses fBlaMka with multiplication):
    fBlaMka(a, b) = a + b + 2*(a mod 2^32)*(b mod 2^32)  mod 2^64

PUBLIC INTERFACE
-----------------
    argon2d (password, salt, time_cost, memory_cost, parallelism, hash_len,
             secret, assoc_data)  -> bytes
    argon2i (...)                 -> bytes
    argon2id(...)                 -> bytes
"""

import sys
import io
import os
import struct
import math

# -- UTF-8 encoding on Windows console ------------------------------------
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# -- import our from-scratch Blake2b --------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from blake2b import blake2b as _blake2b


# =========================================================================
# Constants
# =========================================================================

U64             = 0xFFFFFFFFFFFFFFFF   # 64-bit mask (mod 2^64)
U32             = 0x00000000FFFFFFFF   # 32-bit mask (for fBlaMka)

BLOCK_SIZE      = 1024   # block size in bytes
WORDS_PER_BLOCK = 128    # number of uint64 words per block (1024 / 8)

ARGON2_VERSION  = 19     # algorithm version (0x13)

ARGON2_TYPE_D   = 0      # Argon2d:  data-dependent
ARGON2_TYPE_I   = 1      # Argon2i:  data-independent
ARGON2_TYPE_ID  = 2      # Argon2id: hybrid

SYNC_POINTS     = 4      # number of segments per pass per lane


# =========================================================================
# Step 0 — Block encoding
#
# An Argon2 block = 1024 bytes = 128 64-bit words in little-endian.
# We work on Python lists of integers for readability.
# =========================================================================

def _bytes_to_words(data: bytes) -> list:
    """
    Converts 1024 bytes to a list of 128 64-bit integers (little-endian).

    Raw procedure: read 8 bytes at a time, build the integer manually.
        word = byte[0] + byte[1]*256 + byte[2]*256^2 + ... + byte[7]*256^7
    """
    words = []
    for i in range(WORDS_PER_BLOCK):           # i = 0, 1, ..., 127
        offset = i * 8                         # word position in the block
        word = 0
        for byte_pos in range(8):             # read 8 bytes
            word |= data[offset + byte_pos] << (byte_pos * 8)
        words.append(word)
    return words


def _words_to_bytes(words: list) -> bytes:
    """
    Converts a list of 128 64-bit integers to 1024 bytes (little-endian).

    Raw procedure: write each word byte by byte.
        byte[k] = (word >> (k*8)) & 0xFF
    """
    result = bytearray(BLOCK_SIZE)
    for i in range(WORDS_PER_BLOCK):           # i = 0, 1, ..., 127
        word   = words[i]
        offset = i * 8
        for byte_pos in range(8):
            result[offset + byte_pos] = (word >> (byte_pos * 8)) & 0xFF
    return bytes(result)


def _le32(x: int) -> bytes:
    """Encodes an integer as 4 bytes little-endian."""
    return struct.pack('<I', x)


# =========================================================================
# Step 1 — Mathematical primitives
#
# The basic building block is the G_mix mixing function (8 operations).
# It is DIFFERENT from Blake2b: fBlaMka replaces the simple addition.
# =========================================================================

def _rotr64(x: int, n: int) -> int:
    """
    Right rotation of x on 64 bits by n positions.

    Formula: rotr64(x, n) = (x >> n) | (x << (64-n))  mod 2^64

    Example: rotr64(0x0102030405060708, 8) = 0x0801020304050607
    """
    return ((x >> n) | (x << (64 - n))) & U64


def _fBlaMka(a: int, b: int) -> int:
    """
    Argon2 integer multiplication addition (RFC 9106 §3.4).
    This is the main difference from Blake2b.

    Formula: fBlaMka(a, b) = a + b + 2 * (a mod 2^32) * (b mod 2^32)  mod 2^64

    Why?
    - The term  2*(a mod 2^32)*(b mod 2^32)  adds non-linearity.
    - This makes the compression function harder to implement on FPGA/ASIC.
    - In C: equivalent to  a + b + 2*(uint64_t)(uint32_t)a * (uint64_t)(uint32_t)b
    """
    a_low = a & U32    # keep only the low 32 bits of a
    b_low = b & U32    # keep only the low 32 bits of b
    return (a + b + 2 * a_low * b_low) & U64


def _G_mix(a: int, b: int, c: int, d: int):
    """
    Argon2 G_mix mixing function (8 sequential operations).
    Identical in structure to Blake2b but uses fBlaMka.

    Step 1:  a = fBlaMka(a, b)
    Step 2:  d = rotr64(d XOR a, 32)
    Step 3:  c = fBlaMka(c, d)
    Step 4:  b = rotr64(b XOR c, 24)
    Step 5:  a = fBlaMka(a, b)
    Step 6:  d = rotr64(d XOR a, 16)
    Step 7:  c = fBlaMka(c, d)
    Step 8:  b = rotr64(b XOR c, 63)

    The 4 variables a, b, c, d are mixed through the 8 steps.
    Returns updated (a, b, c, d).
    """
    # Step 1
    a = _fBlaMka(a, b)
    # Step 2
    d = _rotr64(d ^ a, 32)
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


# =========================================================================
# Step 2 — Permutation P on 16 words (BLAKE2_ROUND_NOMSG)
#
# P takes exactly 16 64-bit words.
# They can be visualized as a square of 4 rows x 4 columns:
#
#     v[0]   v[1]   v[2]   v[3]    <- row 0
#     v[4]   v[5]   v[6]   v[7]    <- row 1
#     v[8]   v[9]   v[10]  v[11]   <- row 2
#     v[12]  v[13]  v[14]  v[15]   <- row 3
#
# G_mix is applied on:
#   Columns:   (col0,col1,col2,col3) = (v[0],v[4],v[8],v[12]), etc.
#   Diagonals: (v[0],v[5],v[10],v[15]), etc.  (like Blake2b)
# =========================================================================

def _apply_P(v: list) -> None:
    """
    Argon2 permutation P on a list of 16 uint64 words.
    Modifies the list in place.

    4x4 square structure:
        v[0]  v[1]  v[2]  v[3]
        v[4]  v[5]  v[6]  v[7]
        v[8]  v[9]  v[10] v[11]
        v[12] v[13] v[14] v[15]

    --- 4 G_mix calls on columns ---
    Column 0: (v[0],  v[4],  v[8],  v[12])
    Column 1: (v[1],  v[5],  v[9],  v[13])
    Column 2: (v[2],  v[6],  v[10], v[14])
    Column 3: (v[3],  v[7],  v[11], v[15])

    --- 4 G_mix calls on diagonals ---
    Diagonal 0: (v[0],  v[5],  v[10], v[15])
    Diagonal 1: (v[1],  v[6],  v[11], v[12])
    Diagonal 2: (v[2],  v[7],  v[8],  v[13])
    Diagonal 3: (v[3],  v[4],  v[9],  v[14])
    """
    # --- Columns ---
    v[0],  v[4],  v[8],  v[12] = _G_mix(v[0],  v[4],  v[8],  v[12])
    v[1],  v[5],  v[9],  v[13] = _G_mix(v[1],  v[5],  v[9],  v[13])
    v[2],  v[6],  v[10], v[14] = _G_mix(v[2],  v[6],  v[10], v[14])
    v[3],  v[7],  v[11], v[15] = _G_mix(v[3],  v[7],  v[11], v[15])

    # --- Diagonals ---
    v[0],  v[5],  v[10], v[15] = _G_mix(v[0],  v[5],  v[10], v[15])
    v[1],  v[6],  v[11], v[12] = _G_mix(v[1],  v[6],  v[11], v[12])
    v[2],  v[7],  v[8],  v[13] = _G_mix(v[2],  v[7],  v[8],  v[13])
    v[3],  v[4],  v[9],  v[14] = _G_mix(v[3],  v[4],  v[9],  v[14])


# =========================================================================
# Step 3 — Compression function G(X, Y)  (RFC 9106 §3.4)
#
# Input:  two blocks X and Y (each = list of 128 uint64)
# Output: one block Z (list of 128 uint64)
#
# The 128-word block is viewed as a matrix of 8 rows x 16 words.
# P is applied first on the rows, then on the columns.
#
# Rows    (8 groups of 16 consecutive words):
#   row l = words [ 16*l, 16*l+1, ..., 16*l+15 ]
#
# Columns (8 groups of 16 interleaved words):
#   column l = words [ 2*l, 2*l+1, 2*l+16, 2*l+17, 2*l+32, 2*l+33, ..., 2*l+112, 2*l+113 ]
# =========================================================================

def _compress(X: list, Y: list,
              with_xor: bool = False,
              C_old: list = None) -> list:
    """
    Argon2 compression function G(X, Y) (RFC 9106 §3.4).

    Parameters
    ----------
    X, Y    : input blocks, lists of 128 uint64
    with_xor: True if pass > 0 (XOR with old value)
    C_old   : old content of the slot (used if with_xor=True)

    Step-by-step procedure
    ----------------------
    1. R[i] = X[i] XOR Y[i]          for i = 0..127
    2. Q = copy of R
    3. For each row l (0..7):
         extract  v = Q[16*l .. 16*l+15]
         apply P(v)
         put back into Q
    4. For each column l (0..7):
         extract the 16 interleaved words from Q
         apply P(v)
         put back into Q
    5. Z[i] = Q[i] XOR R[i]          for i = 0..127
    6. If with_xor: Z[i] ^= C_old[i] for i = 0..127
    """

    # --- Step 1: R = X XOR Y ---
    R = [0] * WORDS_PER_BLOCK
    for i in range(WORDS_PER_BLOCK):
        R[i] = X[i] ^ Y[i]

    # --- Step 2: Q = copy of R (working array) ---
    Q = R[:]   # simple copy, Python slice

    # --- Step 3: ROW permutations ---
    # Each row = 16 consecutive words starting at 16*l
    for l in range(8):
        start = l * 16

        # Extract the 16 words of row l
        v = [Q[start + k] for k in range(16)]

        # Apply permutation P
        _apply_P(v)

        # Put the 16 updated words back into Q
        for k in range(16):
            Q[start + k] = v[k]

    # --- Step 4: COLUMN permutations ---
    # Column l = words at positions:
    #   2*l, 2*l+1, 2*l+16, 2*l+17, 2*l+32, 2*l+33, ..., 2*l+112, 2*l+113
    for l in range(8):
        # Compute the 16 indices of column l
        indices = []
        for k in range(8):             # 8 pairs
            indices.append(2*l + k*16)       # even position
            indices.append(2*l + k*16 + 1)   # odd position

        # Extract the 16 words
        v = [Q[idx] for idx in indices]

        # Apply permutation P
        _apply_P(v)

        # Put the 16 updated words back into Q
        for i, idx in enumerate(indices):
            Q[idx] = v[i]

    # --- Step 5: Z = Q XOR R ---
    Z = [0] * WORDS_PER_BLOCK
    for i in range(WORDS_PER_BLOCK):
        Z[i] = Q[i] ^ R[i]

    # --- Step 6 (optional): Z XOR old content (passes > 0) ---
    if with_xor and C_old is not None:
        for i in range(WORDS_PER_BLOCK):
            Z[i] = Z[i] ^ C_old[i]

    return Z


# =========================================================================
# Step 4 — Variable-length hash H'  (RFC 9106 §3.2)
#
# H' is used for two purposes:
#   1. Generate the initial blocks B[i][0] and B[i][1]  (T=1024 bytes)
#   2. Generate the final tag  (T=hash_len bytes)
#
# If T <= 64: simple Blake2b call of length T
# If T >  64: chain of Blake2b calls, collecting 32 bytes per call
# =========================================================================

def _h_prime(data: bytes, T: int) -> bytes:
    """
    Variable-length hash function H'  (RFC 9106 §3.2).

    Parameters
    ----------
    data: data to hash
    T   : desired result length in bytes

    Formula for T <= 64:
        H'(data, T) = Blake2b( LE32(T) || data,  digest_size=T )

    Formula for T > 64:
        r    = ceil(T/32) - 2
        A_1  = Blake2b( LE32(T) || data,  digest_size=64 )
        A_2  = Blake2b( A_1,              digest_size=64 )
        ...
        A_r  = Blake2b( A_{r-1},          digest_size=64 )
        A_{r+1} = Blake2b( A_r,           digest_size = T - 32*r )

        H'(data, T) = A_1[0:32] || A_2[0:32] || ... || A_r[0:32] || A_{r+1}

    Example for T=1024 (block size):
        r = ceil(1024/32) - 2 = 32 - 2 = 30
        Compute 31 Blake2b hashes (30 of 64 bytes + 1 of 64 bytes)
        Take 32 bytes from the first 30 + 64 from the last = 30*32+64 = 1024
    """
    # Mandatory prefix: desired length in little-endian 32 bits
    prefix = _le32(T)

    # Simple case: T fits in a single Blake2b hash
    if T <= 64:
        return _blake2b(prefix + data, digest_size=T)

    # Long case: chain of hashes
    r = math.ceil(T / 32) - 2

    # First hash (includes the LE32(T) prefix and the data)
    A = _blake2b(prefix + data, digest_size=64)

    # Collect the first 32 bytes of each intermediate hash
    result = bytearray()
    result += A[:32]                           # first 32 bytes of A_1

    for _ in range(r - 1):                    # A_2 through A_r
        A = _blake2b(A, digest_size=64)
        result += A[:32]

    # Last hash: variable size to fill exactly T bytes
    last_size = T - 32 * r
    A = _blake2b(A, digest_size=last_size)
    result += A                                # add ALL of A_{r+1}

    return bytes(result)


# =========================================================================
# Step 5 — Reference block selection  (RFC 9106 §3.3)
#
# For each position (pass, lane, segment, index), we compute:
#   J1 = low 32 bits  of the pseudo-random value J
#   J2 = high 32 bits of J
#
# J2 determines the reference LANE.
# J1 determines the reference COLUMN via the phi formula.
#
# The phi formula generates a non-uniform distribution that favors
# recent blocks (the most recent = most likely).
# =========================================================================

def _compute_ref_index(pass_n: int, slice_n: int, idx: int,
                       lane: int, p: int, q: int, segment_len: int,
                       J1: int, same_lane: bool) -> int:
    """
    Computes the column index of the reference block in ref_lane.
    RFC 9106 §3.3, phi formula.

    Parameters
    ----------
    pass_n    : current pass number (0-based)
    slice_n   : current segment number (0..SYNC_POINTS-1)
    idx       : block index within the current segment
    lane      : current lane
    p         : total number of lanes
    q         : number of columns per lane
    segment_len : blocks per segment
    J1        : 32-bit value for indexing
    same_lane : True if the reference lane is the same as the current one

    Returns
    -------
    absolute column index (0..q-1)
    """

    # --- 1. Compute the size of the available reference area ---
    #
    # As the algorithm progresses, more and more blocks
    # become available as references.
    #
    if pass_n == 0:
        # First pass: only already-computed blocks are accessible

        if slice_n == 0:
            # First segment: only blocks before the current index
            # (in the same lane, no blocks in other lanes yet)
            ref_area = idx - 1

        else:
            if same_lane:
                # Same lane: blocks from the start up to the block before the current index
                ref_area = slice_n * segment_len + idx - 1
            else:
                # Other lane: blocks from complete previous segments
                # (the last block of the current segment is not yet available)
                ref_area = slice_n * segment_len - (1 if idx == 0 else 0)

    else:
        # Subsequent passes: almost the entire lane is available
        # (except the current segment being filled)

        if same_lane:
            # Same lane: entire lane except the current segment
            ref_area = q - segment_len + idx - 1
        else:
            # Other lane: entire lane except the current segment
            ref_area = q - segment_len - (1 if idx == 0 else 0)

    # Avoid negative ref_area (edge case at array start)
    if ref_area < 1:
        ref_area = 1

    # --- 2. Phi formula: mapping J1 → index in the reference area ---
    #
    # Non-uniform distribution: recent blocks are more likely.
    #
    # phi(J1) = ref_area - 1 - floor( ref_area * floor(J1^2 / 2^32) / 2^32 )
    #
    # Step by step:
    x = J1                              # J1 is a uint32
    step_a = (x * x) >> 32             # J1^2 divided by 2^32  → uint32
    step_b = (ref_area * step_a) >> 32 # ref_area * step_a / 2^32
    relative_pos = ref_area - 1 - step_b

    # --- 3. Starting position depending on the pass and segment ---
    #
    # The available blocks are not always at the start of the lane.
    # The starting position "rotates" between passes.
    #
    if pass_n == 0:
        # First pass: the reference always starts at the beginning of the lane
        start = 0
    else:
        if slice_n == SYNC_POINTS - 1:
            # Last segment: wrap around to the beginning of the lane
            start = 0
        else:
            # Other segments: start after the current segment
            start = (slice_n + 1) * segment_len

    # --- 4. Absolute index (modulo q to stay within the lane) ---
    ref_col = (start + relative_pos) % q
    return ref_col


# =========================================================================
# Step 6 — Pseudo-random address generation (Argon2i/id)
#
# For Argon2i, J1 and J2 do NOT come from the previous block.
# They are generated via the compression function applied to a block
# built from the current position.
#
# For each segment (pass, lane, slice):
#   input_block = [ pass, lane, slice, m', t, type, counter, 0, 0, ... ]
#   For each batch of 128 values (one block):
#     counter += 1
#     input_block[6] = counter
#     tmp    = G( zeros, input_block )
#     addr   = G( zeros, tmp )
#     J[k]   = addr[k]   for k = 0..127
# =========================================================================

def _generate_addresses(pass_n: int, lane: int, slice_n: int,
                        m_prime: int, t: int, argon2_type: int,
                        segment_len: int) -> list:
    """
    Generates pseudo-random J values for Argon2i/id.
    RFC 9106 §3.3, data-independent addressing.

    Returns a list of segment_len uint64 integers.
    Each integer encodes  J = (J2 << 32) | J1.

    The input block is:
        input[0] = pass    input[1] = lane     input[2] = segment
        input[3] = m'      input[4] = t        input[5] = type
        input[6] = counter (incremented per batch of 128)
        input[7..127] = 0
    """
    # Zero block: 128 words at 0
    zeros = [0] * WORDS_PER_BLOCK

    # Input block with position information
    input_block = [0] * WORDS_PER_BLOCK
    input_block[0] = pass_n
    input_block[1] = lane
    input_block[2] = slice_n
    input_block[3] = m_prime      # total number of blocks
    input_block[4] = t            # number of passes
    input_block[5] = argon2_type  # Argon2 type (1=i, 2=id)
    # input_block[6] = counter (initialized at 0, incremented before use)
    # input_block[7..127] = 0 (already initialized)

    pseudo_rands = []   # list of J values for this segment
    counter      = 0
    addr_block   = None

    for i in range(segment_len):
        # Each batch of WORDS_PER_BLOCK (128) values requires a new computation
        if i % WORDS_PER_BLOCK == 0:
            # Increment the counter and insert it into the input block
            counter += 1
            input_block[6] = counter

            # Two applications of G with a zero block as the first argument
            #   tmp   = G( zeros, input_block )
            #   addr  = G( zeros, tmp )
            tmp        = _compress(zeros, input_block)
            addr_block = _compress(zeros, tmp)

        # The J value for index i is the (i mod 128)-th word of the addr block
        pseudo_rands.append(addr_block[i % WORDS_PER_BLOCK])

    return pseudo_rands


# =========================================================================
# Complete Argon2 algorithm
# =========================================================================

def _argon2(password: bytes, salt: bytes,
            time_cost: int, memory_cost: int, parallelism: int,
            hash_len: int, secret: bytes, assoc_data: bytes,
            argon2_type: int) -> bytes:
    """
    Complete Argon2 implementation (RFC 9106).
    All modes (d, i, id) use this same function.

    Parameters
    ----------
    password   : password P
    salt       : random salt S  (>= 8 bytes recommended)
    time_cost  : number of passes t  (>= 1)
    memory_cost: total memory in KiB m  (>= 8 * parallelism)
    parallelism: number of lanes p  (>= 1)
    hash_len   : output tag length T  (>= 4 bytes)
    secret     : secret key K  (can be empty b'')
    assoc_data : associated data X  (can be empty b'')
    argon2_type: 0=Argon2d, 1=Argon2i, 2=Argon2id

    Returns
    -------
    bytes of length hash_len
    """
    p = parallelism
    T = hash_len
    m = memory_cost
    t = time_cost

    # =================================================================
    # STEP A: Compute m' and the dimensions of the block array
    # =================================================================
    #
    # m' must be divisible by 4*p (4 segments per lane)
    # Round m down to the nearest multiple.
    #
    # m'          = total number of blocks
    # q = m'/p    = columns (blocks) per lane
    # s = q/4     = segment length (blocks per segment per lane)
    #
    m_prime     = max(4 * p, (m // (4 * p)) * (4 * p))
    q           = m_prime // p           # columns per lane
    segment_len = q // SYNC_POINTS       # blocks per segment

    # =================================================================
    # STEP B: Compute H0 (initial hash over all parameters)
    # =================================================================
    #
    # H0 = Blake2b-64(
    #     LE32(p) || LE32(T) || LE32(m) || LE32(t) ||
    #     LE32(version) || LE32(type) ||
    #     LE32(|pwd|)  || pwd  ||
    #     LE32(|salt|) || salt ||
    #     LE32(|K|)    || K    ||
    #     LE32(|X|)    || X    )
    #
    # The "length-prefixed" function = LE32(len) + data
    def len_prefixed(data: bytes) -> bytes:
        return _le32(len(data)) + data

    h0_input = (
        _le32(p)               +   # parallelism
        _le32(T)               +   # tag length
        _le32(m)               +   # requested memory
        _le32(t)               +   # number of passes
        _le32(ARGON2_VERSION)  +   # version 0x13 = 19
        _le32(argon2_type)     +   # type (0=d, 1=i, 2=id)
        len_prefixed(password) +   # password
        len_prefixed(salt)     +   # salt
        len_prefixed(secret)   +   # secret key (optional)
        len_prefixed(assoc_data)   # associated data (optional)
    )
    H0 = _blake2b(h0_input, digest_size=64)

    # =================================================================
    # STEP C: Allocate and initialize memory
    # =================================================================
    #
    # B[lane][col] = a block of 128 uint64 words
    # Organized in p lanes and q columns.
    #
    # Initialization: the first two blocks of each lane
    #   B[i][0] = H'( H0 || LE32(0) || LE32(i),  T=1024 )
    #   B[i][1] = H'( H0 || LE32(1) || LE32(i),  T=1024 )
    #
    B = [[None] * q for _ in range(p)]

    for i in range(p):
        # Block 0 of lane i
        raw_block_0 = _h_prime(H0 + _le32(0) + _le32(i), BLOCK_SIZE)
        B[i][0] = _bytes_to_words(raw_block_0)

        # Block 1 of lane i
        raw_block_1 = _h_prime(H0 + _le32(1) + _le32(i), BLOCK_SIZE)
        B[i][1] = _bytes_to_words(raw_block_1)

    # =================================================================
    # STEP D: Fill loop
    # =================================================================
    #
    # Organization:
    #   t passes (iterations)
    #     4 segments (SYNC_POINTS)
    #       p lanes (parallel in real implementations)
    #         segment_len blocks
    #
    for pass_n in range(t):
        for slice_n in range(SYNC_POINTS):
            for lane in range(p):

                # ---- Determine the addressing mode ----
                #
                # Argon2d:  always data-dependent
                # Argon2i:  always pseudo-random
                # Argon2id: pseudo-random for pass 0 segments 0-1
                #            data-dependent for the rest
                #
                use_independent_addressing = (
                    argon2_type == ARGON2_TYPE_I
                    or
                    (argon2_type == ARGON2_TYPE_ID and pass_n == 0 and slice_n < 2)
                )

                # ---- Pre-compute J values for Argon2i/id ----
                pseudo_rands = None
                if use_independent_addressing:
                    pseudo_rands = _generate_addresses(
                        pass_n, lane, slice_n,
                        m_prime, t, argon2_type,
                        segment_len
                    )

                # ---- Fill each block of the segment ----
                for idx in range(segment_len):
                    col = slice_n * segment_len + idx

                    # Blocks B[i][0] and B[i][1] are already initialized
                    # Skip them only during the first pass of the first segment
                    if pass_n == 0 and slice_n == 0 and idx < 2:
                        continue

                    # ---- Previous block (circular within the lane) ----
                    #
                    # If col == 0, the "previous" is the last block of the lane (col q-1)
                    # This is the circular nature of successive passes.
                    #
                    prev_col   = (col - 1) % q
                    prev_block = B[lane][prev_col]

                    # ---- Obtain the pseudo-random value J ----
                    #
                    # Argon2d: J comes from the first 64 bits of the previous block
                    #           (data-dependent: the choice depends on the data)
                    #
                    # Argon2i: J comes from the pre-computed table
                    #           (data-independent: no information leak)
                    #
                    if use_independent_addressing:
                        J = pseudo_rands[idx]
                    else:
                        J = prev_block[0]   # first word of the previous block

                    # Extract J1 (low 32 bits) and J2 (high 32 bits)
                    J1 = J & U32
                    J2 = (J >> 32) & U32

                    # ---- Determine the reference lane ----
                    #
                    # First pass, first segment: cannot yet reference other lanes
                    # (not yet filled).
                    # Otherwise: J2 mod p gives any lane.
                    #
                    if pass_n == 0 and slice_n == 0:
                        ref_lane = lane           # same lane mandatory
                    else:
                        ref_lane = J2 % p         # any lane

                    same_lane = (ref_lane == lane)

                    # ---- Compute the reference column index ----
                    ref_col = _compute_ref_index(
                        pass_n, slice_n, idx,
                        lane, p, q, segment_len,
                        J1, same_lane
                    )

                    # ---- Apply the compression function ----
                    #
                    # Pass 0: B[lane][col]  = G( prev_block, ref_block )
                    # Pass>0: B[lane][col] ^= G( prev_block, ref_block )
                    #           (XOR with old value to increase diffusion)
                    #
                    with_xor = (pass_n > 0)
                    C_old    = B[lane][col] if with_xor else None

                    B[lane][col] = _compress(
                        prev_block,
                        B[ref_lane][ref_col],
                        with_xor,
                        C_old
                    )

    # =================================================================
    # STEP E: Finalization
    # =================================================================
    #
    # 1. XOR of the last block of each lane
    #    C = B[0][q-1] XOR B[1][q-1] XOR ... XOR B[p-1][q-1]
    #
    # 2. Convert C to bytes, then apply H'(C, T)
    #    to obtain the final tag of length T.
    #
    C = B[0][q - 1][:]               # copy of the last block of lane 0

    for i in range(1, p):
        for w in range(WORDS_PER_BLOCK):
            C[w] = C[w] ^ B[i][q - 1][w]   # XOR word by word

    # Convert the 128 words to bytes for H'
    C_bytes = _words_to_bytes(C)

    # Final tag = H'( C, T )
    tag = _h_prime(C_bytes, T)
    return tag


# =========================================================================
# Public API — three functions, one core
# =========================================================================

def argon2d(password: bytes, salt: bytes,
            time_cost:   int   = 3,
            memory_cost: int   = 65536,
            parallelism: int   = 1,
            hash_len:    int   = 32,
            secret:      bytes = b'',
            assoc_data:  bytes = b'') -> bytes:
    """
    Argon2d  (DATA-DEPENDENT memory access, type=0).

    Used by RandomX (Monero) to build the 256 MiB Cache.
    Data-dependent access makes the construction difficult to parallelize on GPU.

    Parameters
    ----------
    password   : bytes  - the secret to hash (password, key, ...)
    salt       : bytes  - random salt (>= 8 bytes)
    time_cost  : int    - number of passes over memory  (default=3)
    memory_cost: int    - memory in KiB  (default=65536 = 64 MiB)
    parallelism: int    - number of lanes  (default=1)
    hash_len   : int    - result length in bytes  (default=32)
    secret     : bytes  - optional secret key K
    assoc_data : bytes  - optional associated data X

    Returns: bytes of length hash_len
    """
    return _argon2(password, salt, time_cost, memory_cost, parallelism,
                   hash_len, secret, assoc_data, ARGON2_TYPE_D)


def argon2i(password: bytes, salt: bytes,
            time_cost:   int   = 3,
            memory_cost: int   = 65536,
            parallelism: int   = 1,
            hash_len:    int   = 32,
            secret:      bytes = b'',
            assoc_data:  bytes = b'') -> bytes:
    """
    Argon2i  (DATA-INDEPENDENT memory access, type=1).

    The address of each memory access is computed in advance via Blake2b,
    independently of the processed data. This prevents side-channel attacks
    that observe memory accesses.
    Recommended for password hashing.

    Same parameters as argon2d.
    """
    return _argon2(password, salt, time_cost, memory_cost, parallelism,
                   hash_len, secret, assoc_data, ARGON2_TYPE_I)


def argon2id(password: bytes, salt: bytes,
             time_cost:   int   = 3,
             memory_cost: int   = 65536,
             parallelism: int   = 1,
             hash_len:    int   = 32,
             secret:      bytes = b'',
             assoc_data:  bytes = b'') -> bytes:
    """
    Argon2id  (HYBRID d+i, type=2).

    First pass, segments 0 and 1: independent addressing (Argon2i)
    Remaining passes and segments: dependent addressing  (Argon2d)

    Recommended by RFC 9106 as the default mode because it combines
    side-channel resistance (first sweep) and GPU resistance (subsequent passes).

    Same parameters as argon2d.
    """
    return _argon2(password, salt, time_cost, memory_cost, parallelism,
                   hash_len, secret, assoc_data, ARGON2_TYPE_ID)


# =========================================================================
# Self-test  (RFC 9106 Annex B vectors + argon2-cffi verification)
# =========================================================================

if __name__ == '__main__':
    print("=" * 62)
    print("  Argon2 from-scratch  -  self-test")
    print("=" * 62)

    # --- RFC 9106 Annex B vectors (with secret K and data X) ---
    pwd   = bytes([0x01] * 32)
    salt  = bytes([0x02] * 16)
    sec   = bytes([0x03] * 8)
    ad    = bytes([0x04] * 12)
    kw    = dict(time_cost=3, memory_cost=32, parallelism=4, hash_len=32,
                 secret=sec, assoc_data=ad)

    rfc_vectors = [
        ("Argon2d  RFC 9106 B.1", argon2d,
         "512b391b6f1162975371d30919734294f868e3be3984f3c1a13a4db9fabe4acb"),
        ("Argon2id RFC 9106 B.3", argon2id,
         "0d640df58d78766c08c037a34a8b53c9d01ef0452d75b65eb52520e96b01e659"),
    ]

    all_ok = True
    for name, fn, expected in rfc_vectors:
        tag = fn(pwd, salt, **kw)
        ok  = (tag.hex() == expected)
        print(f"\n  [{name}]")
        print(f"  Expected : {expected}")
        print(f"  Got      : {tag.hex()}")
        print(f"  Result   : {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_ok = False

    # --- Optional verification against argon2-cffi ---
    print()
    try:
        from argon2.low_level import hash_secret_raw, Type as Argon2Type
        print("  [argon2-cffi available - additional verification]")
        cffi_pairs = [
            ("Argon2d",  argon2d,  Argon2Type.D),
            ("Argon2i",  argon2i,  Argon2Type.I),
            ("Argon2id", argon2id, Argon2Type.ID),
        ]
        test_pwd  = b"password"
        test_salt = b"saltsalt"
        for name, fn, t in cffi_pairs:
            ref = hash_secret_raw(secret=test_pwd, salt=test_salt,
                                  time_cost=1, memory_cost=8, parallelism=1,
                                  hash_len=32, type=t)
            our = fn(test_pwd, test_salt, time_cost=1, memory_cost=8,
                     parallelism=1, hash_len=32)
            ok  = (ref == our)
            print(f"  {name:<8} vs cffi: {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_ok = False
    except ImportError:
        print("  (argon2-cffi not available, install via: pip install argon2-cffi)")

    print()
    print("=" * 62)
    print(f"  Summary: {'ALL PASS' if all_ok else 'FAILURE(S) DETECTED'}")
    print("=" * 62)
