///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
#include <common/error/cuda_error.hpp>


///////////////////////////////////////////////////////////////////////////////
static constexpr uint32_t ARGON2D_LM5_Q    { 8u   };
static constexpr uint32_t ARGON2D_LM5_WORDS{ 128u };


///////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
uint64_t rotr64_lm5(uint64_t const x, uint32_t const n)
{
    return (x >> n) | (x << (64u - n));
}


///////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void fakeInit_lm5(uint64_t* const out, uint64_t const seed)
{
    #pragma unroll
    for (uint32_t i{ 0u }; i < ARGON2D_LM5_WORDS; ++i)
    {
        out[i] = seed;
    }
}


///////////////////////////////////////////////////////////////////////////////
#define G_MIX_LM5(a, b, c, d)                                                 \
    (a) = (a) + (b) + 2ull * ((a) & 0xFFFFFFFFull) * ((b) & 0xFFFFFFFFull);  \
    (d) = rotr64_lm5((d) ^ (a), 32u);                                         \
    (c) = (c) + (d) + 2ull * ((c) & 0xFFFFFFFFull) * ((d) & 0xFFFFFFFFull);  \
    (b) = rotr64_lm5((b) ^ (c), 24u);                                         \
    (a) = (a) + (b) + 2ull * ((a) & 0xFFFFFFFFull) * ((b) & 0xFFFFFFFFull);  \
    (d) = rotr64_lm5((d) ^ (a), 16u);                                         \
    (c) = (c) + (d) + 2ull * ((c) & 0xFFFFFFFFull) * ((d) & 0xFFFFFFFFull);  \
    (b) = rotr64_lm5((b) ^ (c), 63u);


///////////////////////////////////////////////////////////////////////////////
#define P_LM5(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) \
    G_MIX_LM5(v0,  v4,  v8,  v12)                                             \
    G_MIX_LM5(v1,  v5,  v9,  v13)                                             \
    G_MIX_LM5(v2,  v6,  v10, v14)                                             \
    G_MIX_LM5(v3,  v7,  v11, v15)                                             \
    G_MIX_LM5(v0,  v5,  v10, v15)                                             \
    G_MIX_LM5(v1,  v6,  v11, v12)                                             \
    G_MIX_LM5(v2,  v7,  v8,  v13)                                             \
    G_MIX_LM5(v3,  v4,  v9,  v14)


///////////////////////////////////////////////////////////////////////////////
// R[128] is eliminated: the final write recomputes X[i]^Y[i] instead of
// storing it. Saves 1 KB of local (global) memory per G_block call.
template <bool WITH_XOR>
__device__
void G_block_lm5(
    uint64_t* const       Z,
    uint64_t const* const X,
    uint64_t const* const Y)
{
    uint64_t Q[ARGON2D_LM5_WORDS];

    #pragma unroll
    for (uint32_t i{ 0u }; i < ARGON2D_LM5_WORDS; ++i)
    {
        Q[i] = X[i] ^ Y[i];
    }

    #pragma unroll
    for (uint32_t l{ 0u }; l < 8u; ++l)
    {
        uint64_t v0 { Q[l * 16u +  0u] };
        uint64_t v1 { Q[l * 16u +  1u] };
        uint64_t v2 { Q[l * 16u +  2u] };
        uint64_t v3 { Q[l * 16u +  3u] };
        uint64_t v4 { Q[l * 16u +  4u] };
        uint64_t v5 { Q[l * 16u +  5u] };
        uint64_t v6 { Q[l * 16u +  6u] };
        uint64_t v7 { Q[l * 16u +  7u] };
        uint64_t v8 { Q[l * 16u +  8u] };
        uint64_t v9 { Q[l * 16u +  9u] };
        uint64_t v10{ Q[l * 16u + 10u] };
        uint64_t v11{ Q[l * 16u + 11u] };
        uint64_t v12{ Q[l * 16u + 12u] };
        uint64_t v13{ Q[l * 16u + 13u] };
        uint64_t v14{ Q[l * 16u + 14u] };
        uint64_t v15{ Q[l * 16u + 15u] };

        P_LM5(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)

        Q[l * 16u +  0u] = v0;
        Q[l * 16u +  1u] = v1;
        Q[l * 16u +  2u] = v2;
        Q[l * 16u +  3u] = v3;
        Q[l * 16u +  4u] = v4;
        Q[l * 16u +  5u] = v5;
        Q[l * 16u +  6u] = v6;
        Q[l * 16u +  7u] = v7;
        Q[l * 16u +  8u] = v8;
        Q[l * 16u +  9u] = v9;
        Q[l * 16u + 10u] = v10;
        Q[l * 16u + 11u] = v11;
        Q[l * 16u + 12u] = v12;
        Q[l * 16u + 13u] = v13;
        Q[l * 16u + 14u] = v14;
        Q[l * 16u + 15u] = v15;
    }

    #pragma unroll
    for (uint32_t l{ 0u }; l < 8u; ++l)
    {
        uint64_t v0 { Q[2u * l        ] };
        uint64_t v1 { Q[2u * l +   1u ] };
        uint64_t v2 { Q[2u * l +  16u ] };
        uint64_t v3 { Q[2u * l +  17u ] };
        uint64_t v4 { Q[2u * l +  32u ] };
        uint64_t v5 { Q[2u * l +  33u ] };
        uint64_t v6 { Q[2u * l +  48u ] };
        uint64_t v7 { Q[2u * l +  49u ] };
        uint64_t v8 { Q[2u * l +  64u ] };
        uint64_t v9 { Q[2u * l +  65u ] };
        uint64_t v10{ Q[2u * l +  80u ] };
        uint64_t v11{ Q[2u * l +  81u ] };
        uint64_t v12{ Q[2u * l +  96u ] };
        uint64_t v13{ Q[2u * l +  97u ] };
        uint64_t v14{ Q[2u * l + 112u ] };
        uint64_t v15{ Q[2u * l + 113u ] };

        P_LM5(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)

        Q[2u * l        ] = v0;
        Q[2u * l +   1u ] = v1;
        Q[2u * l +  16u ] = v2;
        Q[2u * l +  17u ] = v3;
        Q[2u * l +  32u ] = v4;
        Q[2u * l +  33u ] = v5;
        Q[2u * l +  48u ] = v6;
        Q[2u * l +  49u ] = v7;
        Q[2u * l +  64u ] = v8;
        Q[2u * l +  65u ] = v9;
        Q[2u * l +  80u ] = v10;
        Q[2u * l +  81u ] = v11;
        Q[2u * l +  96u ] = v12;
        Q[2u * l +  97u ] = v13;
        Q[2u * l + 112u ] = v14;
        Q[2u * l + 113u ] = v15;
    }

    #pragma unroll
    for (uint32_t i{ 0u }; i < ARGON2D_LM5_WORDS; ++i)
    {
        uint64_t z{ Q[i] ^ X[i] ^ Y[i] };
        if (true == WITH_XOR)
        {
            z ^= Z[i];
        }
        Z[i] = z;
    }
}


///////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
uint32_t computeRefColFixed_lm5(
    uint32_t const J1,
    uint32_t const refArea,
    uint32_t const start)
{
    uint64_t const x          { static_cast<uint64_t>(J1) };
    uint64_t const stepA      { (x * x) >> 32u };
    uint64_t const stepB      { (static_cast<uint64_t>(refArea) * stepA) >> 32u };
    uint64_t const relativePos{ static_cast<uint64_t>(refArea) - 1ull - stepB };
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(start) + relativePos) % ARGON2D_LM5_Q);
}


///////////////////////////////////////////////////////////////////////////////
#define ARGON2D_LM5_STEP(prevIdx, colIdx, refArea, start, withXor)            \
{                                                                              \
    uint32_t const J1                                                          \
    {                                                                          \
        static_cast<uint32_t>(B[(prevIdx) * ARGON2D_LM5_WORDS] & 0xFFFFFFFFull) \
    };                                                                         \
    uint32_t const refCol{ computeRefColFixed_lm5(J1, (refArea), (start)) };  \
    G_block_lm5<(withXor)>(                                                    \
        B + (colIdx)  * ARGON2D_LM5_WORDS,                                    \
        B + (prevIdx) * ARGON2D_LM5_WORDS,                                    \
        B + refCol    * ARGON2D_LM5_WORDS);                                   \
}


///////////////////////////////////////////////////////////////////////////////
__global__
void kernel_argon2d_lm5(uint64_t* const memory)
{
    uint32_t const threadId{ blockIdx.x * blockDim.x + threadIdx.x };

    uint64_t* const B
    {
        memory + static_cast<uint64_t>(threadId) * ARGON2D_LM5_Q * ARGON2D_LM5_WORDS
    };

    fakeInit_lm5(B,                        static_cast<uint64_t>(threadId) * 2ull);
    fakeInit_lm5(B + ARGON2D_LM5_WORDS,    static_cast<uint64_t>(threadId) * 2ull + 1ull);

    // Pass 0 — withXor = false, start = 0
    ARGON2D_LM5_STEP(1u, 2u, 1u, 0u, false)
    ARGON2D_LM5_STEP(2u, 3u, 2u, 0u, false)
    ARGON2D_LM5_STEP(3u, 4u, 3u, 0u, false)
    ARGON2D_LM5_STEP(4u, 5u, 4u, 0u, false)
    ARGON2D_LM5_STEP(5u, 6u, 5u, 0u, false)
    ARGON2D_LM5_STEP(6u, 7u, 6u, 0u, false)

    // Pass 1 — withXor = true
    ARGON2D_LM5_STEP(7u, 0u, 5u, 2u, true)
    ARGON2D_LM5_STEP(0u, 1u, 6u, 2u, true)
    ARGON2D_LM5_STEP(1u, 2u, 5u, 4u, true)
    ARGON2D_LM5_STEP(2u, 3u, 6u, 4u, true)
    ARGON2D_LM5_STEP(3u, 4u, 5u, 6u, true)
    ARGON2D_LM5_STEP(4u, 5u, 6u, 6u, true)
    ARGON2D_LM5_STEP(5u, 6u, 5u, 0u, true)
    ARGON2D_LM5_STEP(6u, 7u, 6u, 0u, true)

    // Pass 2 — withXor = true (same pattern as pass 1)
    ARGON2D_LM5_STEP(7u, 0u, 5u, 2u, true)
    ARGON2D_LM5_STEP(0u, 1u, 6u, 2u, true)
    ARGON2D_LM5_STEP(1u, 2u, 5u, 4u, true)
    ARGON2D_LM5_STEP(2u, 3u, 6u, 4u, true)
    ARGON2D_LM5_STEP(3u, 4u, 5u, 6u, true)
    ARGON2D_LM5_STEP(4u, 5u, 6u, 6u, true)
    ARGON2D_LM5_STEP(5u, 6u, 5u, 0u, true)
    ARGON2D_LM5_STEP(6u, 7u, 6u, 0u, true)
}


///////////////////////////////////////////////////////////////////////////////
__host__
bool argon2d_lm5(
    cudaStream_t const stream,
    uint64_t* const    memory,
    uint32_t const     blocks,
    uint32_t const     threads)
{
    kernel_argon2d_lm5<<<blocks, threads, 0, stream>>>(memory);
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
