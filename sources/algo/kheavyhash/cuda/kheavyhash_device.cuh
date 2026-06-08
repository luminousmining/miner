#pragma once

// kHeavyHash (Kaspa) device hash functions for CUDA.
//
// This is a faithful port of sources/algo/kheavyhash/opencl/kheavyhash.cl, which
// is itself KAT-verified bit-identical to the CPU reference. The constants are
// copied verbatim from that kernel (and the CPU reference). It is deliberately
// written in plain C with __CUDACC__ shims so the SAME source can be compiled on
// the host and checked against the known-answer vectors (tests/cuda_device_test.cpp)
// even on a machine without an NVIDIA GPU. Only <cstdint> is required on the host.

#include <cstdint>

#if defined(__CUDACC__)
#define KH_FN __device__ __forceinline__
#define KH_CONST __device__ __constant__
#else
#define KH_FN inline
#define KH_CONST static const
#endif


namespace kheavyhash_cuda
{
    KH_CONST uint64_t POW_INITIAL_STATE[25] = {
        0x113cff0da1f6d83dULL, 0x29bf8855b7027e3cULL, 0x1e5f2e720efb44d2ULL, 0x1ba5a4a3f59869a0ULL,
        0x7b2fafca875e2d65ULL, 0x4aef61d629dce246ULL, 0x183a981ead415b10ULL, 0x776bf60c789bc29cULL,
        0xf8ebf13388663140ULL, 0x2e651c3c43285ff0ULL, 0x0f96070540f14a0aULL, 0x44e367875b299152ULL,
        0xec70f1a425b13715ULL, 0xe6c85d8f82e9da89ULL, 0xb21a601f85b4b223ULL, 0x3485549064a36a46ULL,
        0x0f06dd1c7a2f851aULL, 0xc1a2021d563bb142ULL, 0xba1de5e4451668e4ULL, 0xd102574105095f8dULL,
        0x89ca4e849bcecf4aULL, 0x48b09427a8742edbULL, 0xb1fcce9ce78b5272ULL, 0x5d1129cf82afa5bcULL,
        0x02b97c786f824383ULL };

    KH_CONST uint64_t HEAVY_INITIAL_STATE[25] = {
        0x3ad74c52b2248509ULL, 0x79629b0e2f9f4216ULL, 0x7a14ff4816c7f8eeULL, 0x11a75f4c80056498ULL,
        0xe720e0df44eecedaULL, 0x72c7d82e14f34069ULL, 0xc100ff2a938935baULL, 0x5e219040250fc462ULL,
        0x8039f9a60dcf6a48ULL, 0xa0bcaa9f792a3d0cULL, 0xf431c05dd0a9a226ULL, 0xd31f4cc354c18c3fULL,
        0x6c6b7d01a769cc3dULL, 0x2ec65bd3562493e4ULL, 0x4ef74b3a99cdb044ULL, 0x774c86835434f2b0ULL,
        0x07e961b036bc9416ULL, 0x7e8f1db17765cc07ULL, 0xea8fdb80bac46d39ULL, 0xb992f2d37b34ca58ULL,
        0xc776c5048481b957ULL, 0x47c39f675112c22eULL, 0x92bb399db5290c0aULL, 0x549ae0312f9fc615ULL,
        0x1619327d10b9da35ULL };

    KH_CONST uint64_t ROUND_CONSTANTS[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
        0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL };

    KH_CONST int ROTATIONS[24] = { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
                                   27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44 };

    KH_CONST int PI_LANE[24] = { 10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
                                 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1 };


    KH_FN uint64_t rotl64(uint64_t const x, int const k)
    {
        return (x << k) | (x >> (64 - k));
    }


    KH_FN uint64_t loadLe64(uint8_t const* p)
    {
        uint64_t v = 0;
        for (int b = 0; b < 8; ++b)
        {
            v |= ((uint64_t)p[b]) << (8 * b);
        }
        return v;
    }


    KH_FN void storeLe256(uint64_t const* state, uint8_t* out)
    {
        for (int w = 0; w < 4; ++w)
        {
            for (int b = 0; b < 8; ++b)
            {
                out[w * 8 + b] = (uint8_t)((state[w] >> (8 * b)) & 0xFF);
            }
        }
    }


    KH_FN void keccakF1600(uint64_t* a)
    {
        for (int round = 0; round < 24; ++round)
        {
            uint64_t bc[5];
            for (int i = 0; i < 5; ++i)
            {
                bc[i] = a[i] ^ a[i + 5] ^ a[i + 10] ^ a[i + 15] ^ a[i + 20];
            }
            for (int i = 0; i < 5; ++i)
            {
                uint64_t const t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
                for (int j = 0; j < 25; j += 5)
                {
                    a[j + i] ^= t;
                }
            }

            uint64_t t = a[1];
            for (int i = 0; i < 24; ++i)
            {
                int const      j = PI_LANE[i];
                uint64_t const tmp = a[j];
                a[j] = rotl64(t, ROTATIONS[i]);
                t = tmp;
            }

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

            a[0] ^= ROUND_CONSTANTS[round];
        }
    }


    // hash1 = cSHAKE256("ProofOfWorkHash") over pre_pow | timestamp | zero[32] | nonce.
    KH_FN void powHash(uint8_t const* prePowHash, uint64_t const timestamp, uint64_t const nonce, uint8_t* out)
    {
        uint64_t state[25];
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
    KH_FN void kHeavyHash(uint8_t const* input, uint8_t* out)
    {
        uint64_t state[25];
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


    // Heavy step: matrix * nibble-vector, two rows collapse to one byte (>>10),
    // XOR with hash1, then KHeavyHash. matrix is row-major uint16_t[64*64], 0..15.
    KH_FN void heavyHash(uint16_t const* matrix, uint8_t const* hash1, uint8_t* out)
    {
        uint16_t vec[64];
        for (int i = 0; i < 32; ++i)
        {
            vec[2 * i] = (uint16_t)(hash1[i] >> 4);
            vec[2 * i + 1] = (uint16_t)(hash1[i] & 0x0F);
        }

        uint8_t product[32];
        for (int i = 0; i < 32; ++i)
        {
            uint16_t sum1 = 0;
            uint16_t sum2 = 0;
            for (int j = 0; j < 64; ++j)
            {
                sum1 = (uint16_t)(sum1 + matrix[(2 * i) * 64 + j] * vec[j]);
                sum2 = (uint16_t)(sum2 + matrix[(2 * i + 1) * 64 + j] * vec[j]);
            }
            product[i] = (uint8_t)(((sum1 >> 10) << 4) | (sum2 >> 10));
        }
        for (int i = 0; i < 32; ++i)
        {
            product[i] ^= hash1[i];
        }
        kHeavyHash(product, out);
    }


    // pow <= target as little-endian 256-bit integers (scan from MSB).
    KH_FN bool meetsTarget(uint8_t const* powLe, uint8_t const* targetLe)
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
}
