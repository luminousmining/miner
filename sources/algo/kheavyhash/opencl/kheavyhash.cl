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


// Real mining kernel: each work-item tries nonce = nonceStart + global_id(0).
// On a hit (pow <= target, little-endian) it publishes its nonce.
__kernel void search(__global ushort const* matrix,
                     __global uchar const*  prePowHash,
                     ulong const            timestamp,
                     __global uchar const*  target,
                     ulong const            nonceStart,
                     __global ulong*        foundNonce,
                     __global uint*         foundCount)
{
    ulong const nonce = nonceStart + (ulong)get_global_id(0);

    uchar pre[32];
    uchar tgt[32];
    for (int i = 0; i < 32; ++i)
    {
        pre[i] = prePowHash[i];
        tgt[i] = target[i];
    }

    uchar h1[32];
    powHash(pre, timestamp, nonce, h1);
    uchar pow[32];
    heavyHash(matrix, h1, pow);

    if (meetsTarget(pow, tgt))
    {
        atomic_inc(foundCount);
        foundNonce[0] = nonce;
    }
}
