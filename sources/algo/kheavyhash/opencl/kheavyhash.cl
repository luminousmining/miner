// kHeavyHash (Kaspa) OpenCL kernels.
//
// Correctness-only Layer 3: this kernel must be BIT-IDENTICAL to the CPU
// reference in sources/algo/kheavyhash/*.cpp. The matrix is generated host-side
// once per job (CPU reference generateMatrix) and uploaded; per nonce the kernel
// computes powHash -> heavyHash -> little-endian target compare.
//
// The test_* entry points exist purely so the host KAT harness can check each
// stage against the same known-answer vectors the CPU reference is gated on.
//
// Constants below are copied verbatim from the CPU reference (keccak.cpp,
// hashers.cpp / kheavyhash_test_vectors.hpp). They are NOT re-derived.

__constant ulong POW_INITIAL_STATE[25] = {
    0x113cff0da1f6d83dUL, 0x29bf8855b7027e3cUL, 0x1e5f2e720efb44d2UL, 0x1ba5a4a3f59869a0UL,
    0x7b2fafca875e2d65UL, 0x4aef61d629dce246UL, 0x183a981ead415b10UL, 0x776bf60c789bc29cUL,
    0xf8ebf13388663140UL, 0x2e651c3c43285ff0UL, 0x0f96070540f14a0aUL, 0x44e367875b299152UL,
    0xec70f1a425b13715UL, 0xe6c85d8f82e9da89UL, 0xb21a601f85b4b223UL, 0x3485549064a36a46UL,
    0x0f06dd1c7a2f851aUL, 0xc1a2021d563bb142UL, 0xba1de5e4451668e4UL, 0xd102574105095f8dUL,
    0x89ca4e849bcecf4aUL, 0x48b09427a8742edbUL, 0xb1fcce9ce78b5272UL, 0x5d1129cf82afa5bcUL,
    0x02b97c786f824383UL };

__constant ulong HEAVY_INITIAL_STATE[25] = {
    0x3ad74c52b2248509UL, 0x79629b0e2f9f4216UL, 0x7a14ff4816c7f8eeUL, 0x11a75f4c80056498UL,
    0xe720e0df44eecedaUL, 0x72c7d82e14f34069UL, 0xc100ff2a938935baUL, 0x5e219040250fc462UL,
    0x8039f9a60dcf6a48UL, 0xa0bcaa9f792a3d0cUL, 0xf431c05dd0a9a226UL, 0xd31f4cc354c18c3fUL,
    0x6c6b7d01a769cc3dUL, 0x2ec65bd3562493e4UL, 0x4ef74b3a99cdb044UL, 0x774c86835434f2b0UL,
    0x07e961b036bc9416UL, 0x7e8f1db17765cc07UL, 0xea8fdb80bac46d39UL, 0xb992f2d37b34ca58UL,
    0xc776c5048481b957UL, 0x47c39f675112c22eUL, 0x92bb399db5290c0aUL, 0x549ae0312f9fc615UL,
    0x1619327d10b9da35UL };

__constant ulong ROUND_CONSTANTS[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL, 0x8000000080008000UL,
    0x000000000000808bUL, 0x0000000080000001UL, 0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008aUL, 0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL, 0x000000000000800aUL, 0x800000008000000aUL,
    0x8000000080008081UL, 0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL };

__constant int ROTATIONS[24] = { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
                                 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44 };

__constant int PI_LANE[24] = { 10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
                               15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1 };


inline ulong rotl64(ulong const x, int const k)
{
    return (x << k) | (x >> (64 - k));
}


inline ulong loadLe64(uchar const* p)
{
    ulong v = 0;
    for (int b = 0; b < 8; ++b)
    {
        v |= ((ulong)p[b]) << (8 * b);
    }
    return v;
}


inline void storeLe256(ulong const* state, uchar* out)
{
    for (int w = 0; w < 4; ++w)
    {
        for (int b = 0; b < 8; ++b)
        {
            out[w * 8 + b] = (uchar)((state[w] >> (8 * b)) & 0xFF);
        }
    }
}


void keccakF1600(ulong* a)
{
    for (int round = 0; round < 24; ++round)
    {
        // Theta
        ulong bc[5];
        for (int i = 0; i < 5; ++i)
        {
            bc[i] = a[i] ^ a[i + 5] ^ a[i + 10] ^ a[i + 15] ^ a[i + 20];
        }
        for (int i = 0; i < 5; ++i)
        {
            ulong const t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5)
            {
                a[j + i] ^= t;
            }
        }

        // Rho + Pi
        ulong t = a[1];
        for (int i = 0; i < 24; ++i)
        {
            int const   j = PI_LANE[i];
            ulong const tmp = a[j];
            a[j] = rotl64(t, ROTATIONS[i]);
            t = tmp;
        }

        // Chi
        for (int j = 0; j < 25; j += 5)
        {
            for (int i = 0; i < 5; ++i)
            {
                bc[i] = a[j + i];
            }
            for (int i = 0; i < 5; ++i)
            {
                a[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
            }
        }

        // Iota
        a[0] ^= ROUND_CONSTANTS[round];
    }
}


// hash1 = cSHAKE256("ProofOfWorkHash") over pre_pow_hash | timestamp | zero[32] | nonce.
void powHash(uchar const* prePowHash, ulong const timestamp, ulong const nonce, uchar* out)
{
    ulong state[25];
    for (int i = 0; i < 25; ++i)
    {
        state[i] = POW_INITIAL_STATE[i];
    }
    for (int w = 0; w < 4; ++w)
    {
        state[w] ^= loadLe64(prePowHash + w * 8);
    }
    state[4] ^= timestamp;
    state[9] ^= nonce;
    keccakF1600(state);
    storeLe256(state, out);
}


// hash2 step = cSHAKE256("HeavyHash") over 32 bytes.
void kHeavyHash(uchar const* input, uchar* out)
{
    ulong state[25];
    for (int i = 0; i < 25; ++i)
    {
        state[i] = HEAVY_INITIAL_STATE[i];
    }
    for (int w = 0; w < 4; ++w)
    {
        state[w] ^= loadLe64(input + w * 8);
    }
    keccakF1600(state);
    storeLe256(state, out);
}


// Heavy step: matrix * nibble-vector, collapse two rows to one byte (>>10), XOR
// with hash1, then KHeavyHash. matrix is row-major ushort[64*64], values 0..15.
void heavyHash(__global ushort const* matrix, uchar const* hash1, uchar* out)
{
    ushort vec[64];
    for (int i = 0; i < 32; ++i)
    {
        vec[2 * i] = (ushort)(hash1[i] >> 4);
        vec[2 * i + 1] = (ushort)(hash1[i] & 0x0F);
    }

    uchar product[32];
    for (int i = 0; i < 32; ++i)
    {
        ushort sum1 = 0;
        ushort sum2 = 0;
        for (int j = 0; j < 64; ++j)
        {
            sum1 = (ushort)(sum1 + matrix[(2 * i) * 64 + j] * vec[j]);
            sum2 = (ushort)(sum2 + matrix[(2 * i + 1) * 64 + j] * vec[j]);
        }
        product[i] = (uchar)(((sum1 >> 10) << 4) | (sum2 >> 10));
    }
    for (int i = 0; i < 32; ++i)
    {
        product[i] ^= hash1[i];
    }
    kHeavyHash(product, out);
}


// pow <= target as little-endian 256-bit integers (scan from most-significant byte).
inline bool meetsTarget(uchar const* powLe, uchar const* targetLe)
{
    for (int i = 31; i >= 0; --i)
    {
        if (powLe[i] != targetLe[i])
        {
            return powLe[i] < targetLe[i];
        }
    }
    return true;
}


__kernel void test_pow_hash(__global uchar const* prePowHash,
                            ulong const               timestamp,
                            ulong const               nonce,
                            __global uchar*           out)
{
    uchar pre[32];
    for (int i = 0; i < 32; ++i)
    {
        pre[i] = prePowHash[i];
    }
    uchar h[32];
    powHash(pre, timestamp, nonce, h);
    for (int i = 0; i < 32; ++i)
    {
        out[i] = h[i];
    }
}


__kernel void test_kheavy(__global uchar const* input, __global uchar* out)
{
    uchar in[32];
    for (int i = 0; i < 32; ++i)
    {
        in[i] = input[i];
    }
    uchar h[32];
    kHeavyHash(in, h);
    for (int i = 0; i < 32; ++i)
    {
        out[i] = h[i];
    }
}


__kernel void test_heavy_hash(__global ushort const* matrix,
                              __global uchar const*  hash1,
                              __global uchar*        out)
{
    uchar h1[32];
    for (int i = 0; i < 32; ++i)
    {
        h1[i] = hash1[i];
    }
    uchar h[32];
    heavyHash(matrix, h1, h);
    for (int i = 0; i < 32; ++i)
    {
        out[i] = h[i];
    }
}


// Result buffer shared with the host (mirrors algo::ethash/blake3 Result).
// MAX_RESULT is overridable by the host kernel generator (addDefine).
#ifndef MAX_RESULT
#define MAX_RESULT 4
#endif

typedef struct __attribute__((aligned(8)))
{
    uchar found;
    uint  count;
    ulong nonces[MAX_RESULT];
} Result;


// Real mining kernel: each work-item tries nonce = startNonce + global_id(0).
// On a hit (pow <= target, little-endian) it publishes its nonce into result.
__kernel void kHeavyHash_lm0(__global ushort const* matrix,
                     __global uchar const*  header,
                     __global uchar const*  target,
                     ulong const            timestamp,
                     ulong const            startNonce,
                     __global Result*       result)
{
    ulong const nonce = startNonce + (ulong)get_global_id(0);

    uchar pre[32];
    uchar tgt[32];
    for (int i = 0; i < 32; ++i)
    {
        pre[i] = header[i];
        tgt[i] = target[i];
    }

    uchar h1[32];
    powHash(pre, timestamp, nonce, h1);
    uchar pow[32];
    heavyHash(matrix, h1, pow);

    if (meetsTarget(pow, tgt))
    {
        uint const idx = atomic_inc(&result->count);
        result->found = 1;
        if (idx < MAX_RESULT)
        {
            result->nonces[idx] = nonce;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
// OPTIMIZATION VARIANTS (kHeavyHash_lm1..lm3)
//
// Each is BIT-IDENTICAL to `kHeavyHash_lm0` above (same powHash -> heavyHash -> compare),
// gated by the same KAT vectors. They differ only in HOW the 64x64 nibble matmul
// and keccak are computed:
//   lm1 = stage the matrix in LDS once per workgroup (kill per-nonce global re-read)
//   lm2 = lm1 + integer dot-product matmul (v_dot)
//   lm3 = lm2 + fully register-resident (unrolled rho/pi) keccak
//
// The matrix is uploaded by the host as __global ushort[64*64], values 0..15;
// every variant cooperatively copies it into __local uchar mat[4096] (a byte
// holds 0..15) using a stride loop that is correct for ANY local size, including
// the size-1 launches the KAT uses.
////////////////////////////////////////////////////////////////////////////////

#define KH_MATRIX_N 64
#define KH_MATRIX_ELEMS (KH_MATRIX_N * KH_MATRIX_N)


inline void loadMatrixToLds(__global ushort const* matrix, __local uchar* mat)
{
    uint const lid = (uint)get_local_id(0);
    uint const lsz = (uint)get_local_size(0);
    for (uint i = lid; i < KH_MATRIX_ELEMS; i += lsz)
    {
        mat[i] = (uchar)matrix[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}


inline void publishHit(__global Result* result, ulong const nonce)
{
    uint const idx = atomic_inc(&result->count);
    result->found = 1;
    if (idx < MAX_RESULT)
    {
        result->nonces[idx] = nonce;
    }
}


// ---- lm1: scalar matmul from LDS ------------------------------------------

void heavyHashLds(__local uchar const* matrix, uchar const* hash1, uchar* out)
{
    ushort vec[64];
    for (int i = 0; i < 32; ++i)
    {
        vec[2 * i] = (ushort)(hash1[i] >> 4);
        vec[2 * i + 1] = (ushort)(hash1[i] & 0x0F);
    }

    uchar product[32];
    for (int i = 0; i < 32; ++i)
    {
        ushort sum1 = 0;
        ushort sum2 = 0;
        for (int j = 0; j < 64; ++j)
        {
            sum1 = (ushort)(sum1 + (ushort)matrix[(2 * i) * 64 + j] * vec[j]);
            sum2 = (ushort)(sum2 + (ushort)matrix[(2 * i + 1) * 64 + j] * vec[j]);
        }
        product[i] = (uchar)(((sum1 >> 10) << 4) | (sum2 >> 10));
    }
    for (int i = 0; i < 32; ++i)
    {
        product[i] ^= hash1[i];
    }
    kHeavyHash(product, out);
}


__kernel void kHeavyHash_lm1(__global ushort const* matrix,
                         __global uchar const*  header,
                         __global uchar const*  target,
                         ulong const            timestamp,
                         ulong const            startNonce,
                         __global Result*       result)
{
    __local uchar mat[KH_MATRIX_ELEMS];
    loadMatrixToLds(matrix, mat);

    ulong const nonce = startNonce + (ulong)get_global_id(0);

    uchar pre[32];
    uchar tgt[32];
    for (int i = 0; i < 32; ++i)
    {
        pre[i] = header[i];
        tgt[i] = target[i];
    }

    uchar h1[32];
    powHash(pre, timestamp, nonce, h1);
    uchar pow[32];
    heavyHashLds(mat, h1, pow);

    if (meetsTarget(pow, tgt))
    {
        publishHit(result, nonce);
    }
}


// ---- lm2: integer dot-product matmul from LDS -----------------------------
//
// The 64x64 nibble matmul is reformulated as 4-wide unsigned integer dot
// products. The matrix is staged in LDS packed as uint words (4 column nibbles
// per word, as bytes), the nibble-vector likewise, and each group of 4 MACs
// becomes one v_dot4_u32_u8. 1024 dot ops/nonce vs 4096 scalar MACs.
//
// gfx1201's OpenCL compiler does NOT expose cl_khr_integer_dot_product (verified
// via RGA: neither the macro, dot_4x8packed_*, nor an integer dot() overload
// exist), but the AMD intrinsic __builtin_amdgcn_udot4 lowers directly to
// v_dot4_u32_u8. It is guarded by __AMDGCN__ so POCL / non-AMD ICDs (the CI KAT)
// compile a bit-identical scalar fallback. udot4(m,v,acc) = acc + sum_k mk*vk
// over the 4 bytes, integer-exact, so the result matches lm1 bit-for-bit.

#define KH_MATRIX_WORDS (KH_MATRIX_ELEMS / 4)

#ifdef __AMDGCN__
inline uint khUDot4(uint const m, uint const v, uint const acc)
{
    return __builtin_amdgcn_udot4(m, v, acc, false);
}
#else
inline uint khUDot4(uint const m, uint const v, uint const acc)
{
    return acc + (m & 0xFFu) * (v & 0xFFu) + ((m >> 8) & 0xFFu) * ((v >> 8) & 0xFFu)
           + ((m >> 16) & 0xFFu) * ((v >> 16) & 0xFFu) + ((m >> 24) & 0xFFu) * ((v >> 24) & 0xFFu);
}
#endif


// Pack the host's __global ushort matrix into LDS as uint words: word w holds
// columns [4w..4w+3] of row (w/16) as bytes (LSB = lowest column). Stride loop
// is correct for any local size (incl. the size-1 KAT launch).
inline void loadMatrixToLdsPacked(__global ushort const* matrix, __local uint* matU)
{
    uint const lid = (uint)get_local_id(0);
    uint const lsz = (uint)get_local_size(0);
    for (uint w = lid; w < KH_MATRIX_WORDS; w += lsz)
    {
        uint const base = w * 4u;
        matU[w] = (uint)(uchar)matrix[base] | ((uint)(uchar)matrix[base + 1] << 8)
                  | ((uint)(uchar)matrix[base + 2] << 16) | ((uint)(uchar)matrix[base + 3] << 24);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}


// matrix * nibble-vec via udot4, then XOR with hash1. Writes the 32-byte product
// (the kHeavyHash input). Bit-identical to lm1's heavyHashLds accumulation.
void matmulDot(__local uint const* matU, uchar const* hash1, uchar* product)
{
    uint vecU[16];
    for (int q = 0; q < 16; ++q)
    {
        uint packed = 0;
        for (int k = 0; k < 4; ++k)
        {
            int const   col = 4 * q + k; // 0..63
            uchar const nib = (col & 1) ? (uchar)(hash1[col >> 1] & 0x0F) : (uchar)(hash1[col >> 1] >> 4);
            packed |= (uint)nib << (8 * k);
        }
        vecU[q] = packed;
    }

    for (int i = 0; i < 32; ++i)
    {
        uint sum1 = 0;
        uint sum2 = 0;
        for (int q = 0; q < 16; ++q)
        {
            sum1 = khUDot4(matU[(2 * i) * 16 + q], vecU[q], sum1);
            sum2 = khUDot4(matU[(2 * i + 1) * 16 + q], vecU[q], sum2);
        }
        product[i] = (uchar)(((sum1 >> 10) << 4) | (sum2 >> 10));
    }
    for (int i = 0; i < 32; ++i)
    {
        product[i] ^= hash1[i];
    }
}


__kernel void kHeavyHash_lm2(__global ushort const* matrix,
                         __global uchar const*  header,
                         __global uchar const*  target,
                         ulong const            timestamp,
                         ulong const            startNonce,
                         __global Result*       result)
{
    __local uint matU[KH_MATRIX_WORDS];
    loadMatrixToLdsPacked(matrix, matU);

    ulong const nonce = startNonce + (ulong)get_global_id(0);

    uchar pre[32];
    uchar tgt[32];
    for (int i = 0; i < 32; ++i)
    {
        pre[i] = header[i];
        tgt[i] = target[i];
    }

    uchar h1[32];
    powHash(pre, timestamp, nonce, h1);
    uchar product[32];
    matmulDot(matU, h1, product);
    uchar pow[32];
    kHeavyHash(product, pow);

    if (meetsTarget(pow, tgt))
    {
        publishHit(result, nonce);
    }
}


// ---- lm3: lm2 + register-resident keccak ----------------------------------
//
// The baseline keccakF1600 indexes a private a[25] with a runtime-computed lane
// (PI_LANE[i]) in rho/pi, which forces the array into scratch. Here rho/pi is
// fully unrolled with the PI_LANE/ROTATIONS values as compile-time constants, so
// every access into the state is constant-indexed and the compiler keeps all 25
// lanes in registers. Theta/Chi/Iota are unchanged (already constant-indexed
// after the compiler unrolls their fixed-bound loops). Bit-identical by
// construction (it is the same reference algorithm, unrolled).

inline void keccakF1600U(ulong* a)
{
    for (int round = 0; round < 24; ++round)
    {
        // Theta
        ulong const c0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
        ulong const c1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
        ulong const c2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
        ulong const c3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
        ulong const c4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];
        ulong const d0 = c4 ^ rotl64(c1, 1);
        ulong const d1 = c0 ^ rotl64(c2, 1);
        ulong const d2 = c1 ^ rotl64(c3, 1);
        ulong const d3 = c2 ^ rotl64(c4, 1);
        ulong const d4 = c3 ^ rotl64(c0, 1);
        a[0] ^= d0;  a[5] ^= d0;  a[10] ^= d0; a[15] ^= d0; a[20] ^= d0;
        a[1] ^= d1;  a[6] ^= d1;  a[11] ^= d1; a[16] ^= d1; a[21] ^= d1;
        a[2] ^= d2;  a[7] ^= d2;  a[12] ^= d2; a[17] ^= d2; a[22] ^= d2;
        a[3] ^= d3;  a[8] ^= d3;  a[13] ^= d3; a[18] ^= d3; a[23] ^= d3;
        a[4] ^= d4;  a[9] ^= d4;  a[14] ^= d4; a[19] ^= d4; a[24] ^= d4;

        // Rho + Pi: reference loop unrolled with constant lanes/rotations.
        ulong t = a[1];
        ulong tmp;
        tmp = a[10]; a[10] = rotl64(t, 1);  t = tmp;
        tmp = a[7];  a[7]  = rotl64(t, 3);  t = tmp;
        tmp = a[11]; a[11] = rotl64(t, 6);  t = tmp;
        tmp = a[17]; a[17] = rotl64(t, 10); t = tmp;
        tmp = a[18]; a[18] = rotl64(t, 15); t = tmp;
        tmp = a[3];  a[3]  = rotl64(t, 21); t = tmp;
        tmp = a[5];  a[5]  = rotl64(t, 28); t = tmp;
        tmp = a[16]; a[16] = rotl64(t, 36); t = tmp;
        tmp = a[8];  a[8]  = rotl64(t, 45); t = tmp;
        tmp = a[21]; a[21] = rotl64(t, 55); t = tmp;
        tmp = a[24]; a[24] = rotl64(t, 2);  t = tmp;
        tmp = a[4];  a[4]  = rotl64(t, 14); t = tmp;
        tmp = a[15]; a[15] = rotl64(t, 27); t = tmp;
        tmp = a[23]; a[23] = rotl64(t, 41); t = tmp;
        tmp = a[19]; a[19] = rotl64(t, 56); t = tmp;
        tmp = a[13]; a[13] = rotl64(t, 8);  t = tmp;
        tmp = a[12]; a[12] = rotl64(t, 25); t = tmp;
        tmp = a[2];  a[2]  = rotl64(t, 43); t = tmp;
        tmp = a[20]; a[20] = rotl64(t, 62); t = tmp;
        tmp = a[14]; a[14] = rotl64(t, 18); t = tmp;
        tmp = a[22]; a[22] = rotl64(t, 39); t = tmp;
        tmp = a[9];  a[9]  = rotl64(t, 61); t = tmp;
        tmp = a[6];  a[6]  = rotl64(t, 20); t = tmp;
        tmp = a[1];  a[1]  = rotl64(t, 44); t = tmp;

        // Chi
        for (int j = 0; j < 25; j += 5)
        {
            ulong const b0 = a[j + 0];
            ulong const b1 = a[j + 1];
            ulong const b2 = a[j + 2];
            ulong const b3 = a[j + 3];
            ulong const b4 = a[j + 4];
            a[j + 0] ^= (~b1) & b2;
            a[j + 1] ^= (~b2) & b3;
            a[j + 2] ^= (~b3) & b4;
            a[j + 3] ^= (~b4) & b0;
            a[j + 4] ^= (~b0) & b1;
        }

        // Iota
        a[0] ^= ROUND_CONSTANTS[round];
    }
}


void powHashU(uchar const* prePowHash, ulong const timestamp, ulong const nonce, uchar* out)
{
    ulong state[25];
    for (int i = 0; i < 25; ++i)
    {
        state[i] = POW_INITIAL_STATE[i];
    }
    for (int w = 0; w < 4; ++w)
    {
        state[w] ^= loadLe64(prePowHash + w * 8);
    }
    state[4] ^= timestamp;
    state[9] ^= nonce;
    keccakF1600U(state);
    storeLe256(state, out);
}


void kHeavyHashU(uchar const* input, uchar* out)
{
    ulong state[25];
    for (int i = 0; i < 25; ++i)
    {
        state[i] = HEAVY_INITIAL_STATE[i];
    }
    for (int w = 0; w < 4; ++w)
    {
        state[w] ^= loadLe64(input + w * 8);
    }
    keccakF1600U(state);
    storeLe256(state, out);
}


__kernel void kHeavyHash_lm3(__global ushort const* matrix,
                         __global uchar const*  header,
                         __global uchar const*  target,
                         ulong const            timestamp,
                         ulong const            startNonce,
                         __global Result*       result)
{
    __local uint matU[KH_MATRIX_WORDS];
    loadMatrixToLdsPacked(matrix, matU);

    ulong const nonce = startNonce + (ulong)get_global_id(0);

    uchar pre[32];
    uchar tgt[32];
    for (int i = 0; i < 32; ++i)
    {
        pre[i] = header[i];
        tgt[i] = target[i];
    }

    uchar h1[32];
    powHashU(pre, timestamp, nonce, h1);
    uchar product[32];
    matmulDot(matU, h1, product);
    uchar pow[32];
    kHeavyHashU(product, pow);

    if (meetsTarget(pow, tgt))
    {
        publishHit(result, nonce);
    }
}


// ---- lm4: lm2 + powHash keccak midstate -----------------------------------
//
// In powHash only state[9] (nonce) varies per nonce; state[0..3] (prePowHash)
// and state[4] (timestamp) are fixed for the whole job. So the FIRST keccak's
// round-1 theta is almost entirely job-constant: compute the post-theta state
// once per workgroup (thread 0 -> LDS) and per nonce only fold the nonce back
// in. The nonce enters round-1 theta in exactly two ways: directly via state[9]
// (column 4), and via the column-4 parity c4 -> d0 (column 0) and d3 (column 3,
// as rotl(nonce,1)). So starting from the nonce-free post-theta midstate, per
// nonce: XOR nonce into lanes {0,5,10,15,20,9} and rotl(nonce,1) into lanes
// {3,8,13,18,23}, then run keccak from round-1 rho/pi onward. The heavy (second)
// keccak depends on the matmul output and cannot be hoisted, so it is full.
// Bit-identical to lm2 (KAT-gated); only the first keccak's round-1 theta moves.

inline void khTheta(ulong* a)
{
    ulong const c0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
    ulong const c1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
    ulong const c2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
    ulong const c3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
    ulong const c4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];
    ulong const d0 = c4 ^ rotl64(c1, 1);
    ulong const d1 = c0 ^ rotl64(c2, 1);
    ulong const d2 = c1 ^ rotl64(c3, 1);
    ulong const d3 = c2 ^ rotl64(c4, 1);
    ulong const d4 = c3 ^ rotl64(c0, 1);
    a[0] ^= d0;  a[5] ^= d0;  a[10] ^= d0; a[15] ^= d0; a[20] ^= d0;
    a[1] ^= d1;  a[6] ^= d1;  a[11] ^= d1; a[16] ^= d1; a[21] ^= d1;
    a[2] ^= d2;  a[7] ^= d2;  a[12] ^= d2; a[17] ^= d2; a[22] ^= d2;
    a[3] ^= d3;  a[8] ^= d3;  a[13] ^= d3; a[18] ^= d3; a[23] ^= d3;
    a[4] ^= d4;  a[9] ^= d4;  a[14] ^= d4; a[19] ^= d4; a[24] ^= d4;
}


inline void khRhoPi(ulong* a)
{
    ulong t = a[1];
    ulong tmp;
    tmp = a[10]; a[10] = rotl64(t, 1);  t = tmp;
    tmp = a[7];  a[7]  = rotl64(t, 3);  t = tmp;
    tmp = a[11]; a[11] = rotl64(t, 6);  t = tmp;
    tmp = a[17]; a[17] = rotl64(t, 10); t = tmp;
    tmp = a[18]; a[18] = rotl64(t, 15); t = tmp;
    tmp = a[3];  a[3]  = rotl64(t, 21); t = tmp;
    tmp = a[5];  a[5]  = rotl64(t, 28); t = tmp;
    tmp = a[16]; a[16] = rotl64(t, 36); t = tmp;
    tmp = a[8];  a[8]  = rotl64(t, 45); t = tmp;
    tmp = a[21]; a[21] = rotl64(t, 55); t = tmp;
    tmp = a[24]; a[24] = rotl64(t, 2);  t = tmp;
    tmp = a[4];  a[4]  = rotl64(t, 14); t = tmp;
    tmp = a[15]; a[15] = rotl64(t, 27); t = tmp;
    tmp = a[23]; a[23] = rotl64(t, 41); t = tmp;
    tmp = a[19]; a[19] = rotl64(t, 56); t = tmp;
    tmp = a[13]; a[13] = rotl64(t, 8);  t = tmp;
    tmp = a[12]; a[12] = rotl64(t, 25); t = tmp;
    tmp = a[2];  a[2]  = rotl64(t, 43); t = tmp;
    tmp = a[20]; a[20] = rotl64(t, 62); t = tmp;
    tmp = a[14]; a[14] = rotl64(t, 18); t = tmp;
    tmp = a[22]; a[22] = rotl64(t, 39); t = tmp;
    tmp = a[9];  a[9]  = rotl64(t, 61); t = tmp;
    tmp = a[6];  a[6]  = rotl64(t, 20); t = tmp;
    tmp = a[1];  a[1]  = rotl64(t, 44); t = tmp;
}


inline void khChi(ulong* a)
{
    for (int j = 0; j < 25; j += 5)
    {
        ulong const b0 = a[j + 0];
        ulong const b1 = a[j + 1];
        ulong const b2 = a[j + 2];
        ulong const b3 = a[j + 3];
        ulong const b4 = a[j + 4];
        a[j + 0] ^= (~b1) & b2;
        a[j + 1] ^= (~b2) & b3;
        a[j + 2] ^= (~b3) & b4;
        a[j + 3] ^= (~b4) & b0;
        a[j + 4] ^= (~b0) & b1;
    }
}


// keccak-f[1600] assuming round-0 THETA has already been applied to `a`.
inline void keccakF1600FromTheta1(ulong* a)
{
    khRhoPi(a);
    khChi(a);
    a[0] ^= ROUND_CONSTANTS[0];
    for (int round = 1; round < 24; ++round)
    {
        khTheta(a);
        khRhoPi(a);
        khChi(a);
        a[0] ^= ROUND_CONSTANTS[round];
    }
}


__kernel void kHeavyHash_lm4(__global ushort const* matrix,
                         __global uchar const*  header,
                         __global uchar const*  target,
                         ulong const            timestamp,
                         ulong const            startNonce,
                         __global Result*       result)
{
    __local uint  matU[KH_MATRIX_WORDS];
    __local ulong powMid[25]; // powHash state after round-1 theta, nonce excluded

    uint const lid = (uint)get_local_id(0);
    uint const lsz = (uint)get_local_size(0);
    for (uint w = lid; w < KH_MATRIX_WORDS; w += lsz)
    {
        uint const base = w * 4u;
        matU[w] = (uint)(uchar)matrix[base] | ((uint)(uchar)matrix[base + 1] << 8)
                  | ((uint)(uchar)matrix[base + 2] << 16) | ((uint)(uchar)matrix[base + 3] << 24);
    }
    if (0u == lid)
    {
        uchar pre[32];
        for (int i = 0; i < 32; ++i)
        {
            pre[i] = header[i];
        }
        ulong ms[25];
        for (int i = 0; i < 25; ++i)
        {
            ms[i] = POW_INITIAL_STATE[i];
        }
        for (int w = 0; w < 4; ++w)
        {
            ms[w] ^= loadLe64(pre + w * 8);
        }
        ms[4] ^= timestamp; // nonce (ms[9]) intentionally NOT folded in here
        khTheta(ms);
        for (int i = 0; i < 25; ++i)
        {
            powMid[i] = ms[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    ulong const nonce = startNonce + (ulong)get_global_id(0);

    uchar tgt[32];
    for (int i = 0; i < 32; ++i)
    {
        tgt[i] = target[i];
    }

    // powHash from the shared midstate: fold this nonce back into round-1 theta.
    ulong st[25];
    for (int i = 0; i < 25; ++i)
    {
        st[i] = powMid[i];
    }
    ulong const nr = rotl64(nonce, 1);
    st[0] ^= nonce; st[5] ^= nonce; st[10] ^= nonce; st[15] ^= nonce; st[20] ^= nonce; st[9] ^= nonce;
    st[3] ^= nr;    st[8] ^= nr;    st[13] ^= nr;    st[18] ^= nr;    st[23] ^= nr;
    keccakF1600FromTheta1(st);
    uchar h1[32];
    storeLe256(st, h1);

    uchar product[32];
    matmulDot(matU, h1, product);
    uchar pow[32];
    kHeavyHash(product, pow);

    if (meetsTarget(pow, tgt))
    {
        publishHit(result, nonce);
    }
}


// ---- lm5: lm4 + heavy (second) keccak round-0 constant-parity hoist --------
//
// The heavy keccak absorbs only 32 bytes (the matmul product) into lanes 0..3 of
// HEAVY_INITIAL_STATE; lanes 4..24 stay at the compile-time constant base state.
// So its round-0 theta is the ONLY hoistable part: column parity c4 is fully
// constant and c0..c3 collapse to (input_lane ^ const). Unlike powHash's nonce
// (which touches a single column, letting the WHOLE round-0 theta hoist in lm4),
// the product fills 4 of 5 columns, so every theta D-value depends on the input
// and nothing hoists past round-0 theta -- chi is nonlinear and blocks it. This
// trims round-0 parity reduction from 20 XORs to 4 (the constants fold at
// compile time from the __constant base state). Bit-identical to lm4 (KAT-gated);
// the first keccak and matmul are identical to lm4, only the heavy keccak's
// round-0 theta is specialised.

inline void kHeavyHashHoisted(uchar const* input, uchar* out)
{
    // Absorb: lanes 0..3 = base ^ input word; lanes 4..24 = base (constant).
    ulong const a0 = HEAVY_INITIAL_STATE[0] ^ loadLe64(input + 0);
    ulong const a1 = HEAVY_INITIAL_STATE[1] ^ loadLe64(input + 8);
    ulong const a2 = HEAVY_INITIAL_STATE[2] ^ loadLe64(input + 16);
    ulong const a3 = HEAVY_INITIAL_STATE[3] ^ loadLe64(input + 24);

    // Round-0 theta. Constant column reductions (lanes 4..24 fixed) fold away;
    // only c0..c3 carry one input lane each, c4 is fully constant.
    ulong const c0 = a0 ^ (HEAVY_INITIAL_STATE[5] ^ HEAVY_INITIAL_STATE[10] ^ HEAVY_INITIAL_STATE[15]
                           ^ HEAVY_INITIAL_STATE[20]);
    ulong const c1 = a1 ^ (HEAVY_INITIAL_STATE[6] ^ HEAVY_INITIAL_STATE[11] ^ HEAVY_INITIAL_STATE[16]
                           ^ HEAVY_INITIAL_STATE[21]);
    ulong const c2 = a2 ^ (HEAVY_INITIAL_STATE[7] ^ HEAVY_INITIAL_STATE[12] ^ HEAVY_INITIAL_STATE[17]
                           ^ HEAVY_INITIAL_STATE[22]);
    ulong const c3 = a3 ^ (HEAVY_INITIAL_STATE[8] ^ HEAVY_INITIAL_STATE[13] ^ HEAVY_INITIAL_STATE[18]
                           ^ HEAVY_INITIAL_STATE[23]);
    ulong const c4 = HEAVY_INITIAL_STATE[4] ^ HEAVY_INITIAL_STATE[9] ^ HEAVY_INITIAL_STATE[14]
                     ^ HEAVY_INITIAL_STATE[19] ^ HEAVY_INITIAL_STATE[24];
    ulong const d0 = c4 ^ rotl64(c1, 1);
    ulong const d1 = c0 ^ rotl64(c2, 1);
    ulong const d2 = c1 ^ rotl64(c3, 1);
    ulong const d3 = c2 ^ rotl64(c4, 1); // rotl64(c4,1) is compile-constant
    ulong const d4 = c3 ^ rotl64(c0, 1);

    ulong a[25];
    a[0] = a0 ^ d0;
    a[1] = a1 ^ d1;
    a[2] = a2 ^ d2;
    a[3] = a3 ^ d3;
    a[4] = HEAVY_INITIAL_STATE[4] ^ d4;
    a[5] = HEAVY_INITIAL_STATE[5] ^ d0;
    a[6] = HEAVY_INITIAL_STATE[6] ^ d1;
    a[7] = HEAVY_INITIAL_STATE[7] ^ d2;
    a[8] = HEAVY_INITIAL_STATE[8] ^ d3;
    a[9] = HEAVY_INITIAL_STATE[9] ^ d4;
    a[10] = HEAVY_INITIAL_STATE[10] ^ d0;
    a[11] = HEAVY_INITIAL_STATE[11] ^ d1;
    a[12] = HEAVY_INITIAL_STATE[12] ^ d2;
    a[13] = HEAVY_INITIAL_STATE[13] ^ d3;
    a[14] = HEAVY_INITIAL_STATE[14] ^ d4;
    a[15] = HEAVY_INITIAL_STATE[15] ^ d0;
    a[16] = HEAVY_INITIAL_STATE[16] ^ d1;
    a[17] = HEAVY_INITIAL_STATE[17] ^ d2;
    a[18] = HEAVY_INITIAL_STATE[18] ^ d3;
    a[19] = HEAVY_INITIAL_STATE[19] ^ d4;
    a[20] = HEAVY_INITIAL_STATE[20] ^ d0;
    a[21] = HEAVY_INITIAL_STATE[21] ^ d1;
    a[22] = HEAVY_INITIAL_STATE[22] ^ d2;
    a[23] = HEAVY_INITIAL_STATE[23] ^ d3;
    a[24] = HEAVY_INITIAL_STATE[24] ^ d4;

    // Finish round 0 (rho/pi, chi, iota), then rounds 1..23.
    khRhoPi(a);
    khChi(a);
    a[0] ^= ROUND_CONSTANTS[0];
    for (int round = 1; round < 24; ++round)
    {
        khTheta(a);
        khRhoPi(a);
        khChi(a);
        a[0] ^= ROUND_CONSTANTS[round];
    }
    storeLe256(a, out);
}


__kernel void kHeavyHash_lm5(__global ushort const* matrix,
                         __global uchar const*  header,
                         __global uchar const*  target,
                         ulong const            timestamp,
                         ulong const            startNonce,
                         __global Result*       result)
{
    __local uint  matU[KH_MATRIX_WORDS];
    __local ulong powMid[25]; // powHash state after round-1 theta, nonce excluded

    uint const lid = (uint)get_local_id(0);
    uint const lsz = (uint)get_local_size(0);
    for (uint w = lid; w < KH_MATRIX_WORDS; w += lsz)
    {
        uint const base = w * 4u;
        matU[w] = (uint)(uchar)matrix[base] | ((uint)(uchar)matrix[base + 1] << 8)
                  | ((uint)(uchar)matrix[base + 2] << 16) | ((uint)(uchar)matrix[base + 3] << 24);
    }
    if (0u == lid)
    {
        uchar pre[32];
        for (int i = 0; i < 32; ++i)
        {
            pre[i] = header[i];
        }
        ulong ms[25];
        for (int i = 0; i < 25; ++i)
        {
            ms[i] = POW_INITIAL_STATE[i];
        }
        for (int w = 0; w < 4; ++w)
        {
            ms[w] ^= loadLe64(pre + w * 8);
        }
        ms[4] ^= timestamp;
        khTheta(ms);
        for (int i = 0; i < 25; ++i)
        {
            powMid[i] = ms[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    ulong const nonce = startNonce + (ulong)get_global_id(0);

    uchar tgt[32];
    for (int i = 0; i < 32; ++i)
    {
        tgt[i] = target[i];
    }

    ulong st[25];
    for (int i = 0; i < 25; ++i)
    {
        st[i] = powMid[i];
    }
    ulong const nr = rotl64(nonce, 1);
    st[0] ^= nonce; st[5] ^= nonce; st[10] ^= nonce; st[15] ^= nonce; st[20] ^= nonce; st[9] ^= nonce;
    st[3] ^= nr;    st[8] ^= nr;    st[13] ^= nr;    st[18] ^= nr;    st[23] ^= nr;
    keccakF1600FromTheta1(st);
    uchar h1[32];
    storeLe256(st, h1);

    uchar product[32];
    matmulDot(matU, h1, product);
    uchar pow[32];
    kHeavyHashHoisted(product, pow);

    if (meetsTarget(pow, tgt))
    {
        publishHit(result, nonce);
    }
}
