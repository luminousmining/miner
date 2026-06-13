// kHeavyHash (Kaspa) OpenCL benchmark kernel -- lm2 (v_dot4 packed matmul).
//
// Correctness-only Layer 3: bit-identical to the CPU reference in
// sources/algo/kheavyhash/*.cpp. Same pipeline as lm0; the heavy-step matmul is
// done with a 4-wide udot4 over a byte-packed matrix staged in local memory.
//
// Constants below are copied verbatim from the CPU reference (keccak.cpp,
// hashers.cpp). They are NOT re-derived.

#include "kernel/common/rotate_byte.cl"
#include "kernel/common/xor.cl"
#include "kernel/common/load_store_le.cl"
#include "kernel/crypto/keccak_f1600.cl"
#include "kernel/common/result.cl"


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


// Full Keccak-f[1600] permutation, reusing the shared round in
// kernel/crypto/keccak_f1600.cl so the round logic is not duplicated here.
inline void keccakF1600(ulong* state)
{
    for (int round = 0; round < 24; ++round)
    {
        keccak_f1600_round(state, ROUND_CONSTANTS[round]);
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
        state[w] ^= load_le_u64(prePowHash + w * 8);
    }
    state[4] ^= timestamp;
    state[9] ^= nonce;
    keccakF1600(state);
    store_le_u256(state, out);
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
        state[w] ^= load_le_u64(input + w * 8);
    }
    keccakF1600(state);
    store_le_u256(state, out);
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


#define KH_MATRIX_N 64


#define KH_MATRIX_ELEMS (KH_MATRIX_N * KH_MATRIX_N)


inline void publishHit(__global t_result* result, ulong const nonce)
{
    atomic_inc(&result->count);
    result->found = true;
    result->nonce = nonce;
}


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


// Real mining kernel: each work-item tries nonce = startNonce + global_id(0).
// On a hit (pow <= target, little-endian) it publishes its nonce into result.
__kernel void kHeavyHash_lm2(__global ushort const* matrix,
                             __global uchar const*  header,
                             __global uchar const*  target,
                             ulong const            timestamp,
                             ulong const            startNonce,
                             __global t_result*     result)
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
