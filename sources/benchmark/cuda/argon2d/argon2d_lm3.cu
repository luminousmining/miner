///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
#include <common/error/cuda_error.hpp>


///////////////////////////////////////////////////////////////////////////////
static constexpr uint32_t ARGON2D_LM3_Q    { 8u   };
static constexpr uint32_t ARGON2D_LM3_WORDS{ 128u };


///////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
uint64_t rotr64_lm3(uint64_t const x, uint32_t const n)
{
    return (x >> n) | (x << (64u - n));
}


///////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void fakeInit_lm3(uint64_t* const out, uint64_t const seed)
{
    #pragma unroll
    for (uint32_t i{ 0u }; i < ARGON2D_LM3_WORDS; ++i)
    {
        out[i] = seed;
    }
}


///////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void G_mix_lm3(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d)
{
    a = a + b + 2ull * (a & 0xFFFFFFFFull) * (b & 0xFFFFFFFFull);
    d = rotr64_lm3(d ^ a, 32u);
    c = c + d + 2ull * (c & 0xFFFFFFFFull) * (d & 0xFFFFFFFFull);
    b = rotr64_lm3(b ^ c, 24u);
    a = a + b + 2ull * (a & 0xFFFFFFFFull) * (b & 0xFFFFFFFFull);
    d = rotr64_lm3(d ^ a, 16u);
    c = c + d + 2ull * (c & 0xFFFFFFFFull) * (d & 0xFFFFFFFFull);
    b = rotr64_lm3(b ^ c, 63u);
}


///////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void P_lm3(uint64_t* const v)
{
    G_mix_lm3(v[0],  v[4],  v[8],  v[12]);
    G_mix_lm3(v[1],  v[5],  v[9],  v[13]);
    G_mix_lm3(v[2],  v[6],  v[10], v[14]);
    G_mix_lm3(v[3],  v[7],  v[11], v[15]);

    G_mix_lm3(v[0],  v[5],  v[10], v[15]);
    G_mix_lm3(v[1],  v[6],  v[11], v[12]);
    G_mix_lm3(v[2],  v[7],  v[8],  v[13]);
    G_mix_lm3(v[3],  v[4],  v[9],  v[14]);
}


///////////////////////////////////////////////////////////////////////////////
// G_block_lm3: template eliminates the withXor branch at compile time.
// Instantiated as G_block_lm3<false> (pass 0) and G_block_lm3<true> (pass 1+).
template <bool WITH_XOR>
__device__
void G_block_lm3(
    uint64_t* const       Z,
    uint64_t const* const X,
    uint64_t const* const Y)
{
    uint64_t R[ARGON2D_LM3_WORDS];
    uint64_t Q[ARGON2D_LM3_WORDS];

    #pragma unroll
    for (uint32_t i{ 0u }; i < ARGON2D_LM3_WORDS; ++i)
    {
        R[i] = X[i] ^ Y[i];
        Q[i] = R[i];
    }

    #pragma unroll
    for (uint32_t l{ 0u }; l < 8u; ++l)
    {
        P_lm3(Q + l * 16u);
    }

    #pragma unroll
    for (uint32_t l{ 0u }; l < 8u; ++l)
    {
        uint64_t v[16];

        #pragma unroll
        for (uint32_t k{ 0u }; k < 8u; ++k)
        {
            v[k * 2u]      = Q[2u * l + k * 16u];
            v[k * 2u + 1u] = Q[2u * l + k * 16u + 1u];
        }

        P_lm3(v);

        #pragma unroll
        for (uint32_t k{ 0u }; k < 8u; ++k)
        {
            Q[2u * l + k * 16u]      = v[k * 2u];
            Q[2u * l + k * 16u + 1u] = v[k * 2u + 1u];
        }
    }

    #pragma unroll
    for (uint32_t i{ 0u }; i < ARGON2D_LM3_WORDS; ++i)
    {
        uint64_t z{ Q[i] ^ R[i] };
        if (true == WITH_XOR)
        {
            z ^= Z[i];
        }
        Z[i] = z;
    }
}


///////////////////////////////////////////////////////////////////////////////
// computeRefColFixed_lm3: refArea and start are compile-time constants
// when called from the unrolled fill loop, enabling full phi-formula folding.
__device__ __forceinline__
uint32_t computeRefColFixed_lm3(
    uint32_t const J1,
    uint32_t const refArea,
    uint32_t const start)
{
    uint64_t const x          { static_cast<uint64_t>(J1) };
    uint64_t const stepA      { (x * x) >> 32u };
    uint64_t const stepB      { (static_cast<uint64_t>(refArea) * stepA) >> 32u };
    uint64_t const relativePos{ static_cast<uint64_t>(refArea) - 1ull - stepB };
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(start) + relativePos) % ARGON2D_LM3_Q);
}


///////////////////////////////////////////////////////////////////////////////
// ARGON2D_LM3_STEP: one explicit G_block call with all parameters hardcoded.
// withXor must be a compile-time boolean literal (false or true).
#define ARGON2D_LM3_STEP(prevIdx, colIdx, refArea, start, withXor)            \
{                                                                              \
    uint32_t const J1                                                          \
    {                                                                          \
        static_cast<uint32_t>(B[(prevIdx) * ARGON2D_LM3_WORDS] & 0xFFFFFFFFull) \
    };                                                                         \
    uint32_t const refCol{ computeRefColFixed_lm3(J1, (refArea), (start)) };  \
    G_block_lm3<(withXor)>(                                                    \
        B + (colIdx)  * ARGON2D_LM3_WORDS,                                    \
        B + (prevIdx) * ARGON2D_LM3_WORDS,                                    \
        B + refCol    * ARGON2D_LM3_WORDS);                                   \
}


///////////////////////////////////////////////////////////////////////////////
// Fill loop fully unrolled: 22 explicit G_block calls (6 for pass 0,
// 8 for pass 1, 8 for pass 2). The continue and loop overhead are gone.
// Calling convention: ARGON2D_LM3_STEP(prevCol, col, refArea, start, withXor)
__global__
void kernel_argon2d_lm3(uint64_t* const memory)
{
    uint32_t const threadId{ blockIdx.x * blockDim.x + threadIdx.x };

    uint64_t* const B
    {
        memory + static_cast<uint64_t>(threadId) * ARGON2D_LM3_Q * ARGON2D_LM3_WORDS
    };

    fakeInit_lm3(B,                        static_cast<uint64_t>(threadId) * 2ull);
    fakeInit_lm3(B + ARGON2D_LM3_WORDS,    static_cast<uint64_t>(threadId) * 2ull + 1ull);

    // Pass 0 — withXor = false, start = 0
    ARGON2D_LM3_STEP(1u, 2u, 1u, 0u, false)    // slice=1 idx=0
    ARGON2D_LM3_STEP(2u, 3u, 2u, 0u, false)    // slice=1 idx=1
    ARGON2D_LM3_STEP(3u, 4u, 3u, 0u, false)    // slice=2 idx=0
    ARGON2D_LM3_STEP(4u, 5u, 4u, 0u, false)    // slice=2 idx=1
    ARGON2D_LM3_STEP(5u, 6u, 5u, 0u, false)    // slice=3 idx=0
    ARGON2D_LM3_STEP(6u, 7u, 6u, 0u, false)    // slice=3 idx=1

    // Pass 1 — withXor = true
    ARGON2D_LM3_STEP(7u, 0u, 5u, 2u, true)     // slice=0 idx=0
    ARGON2D_LM3_STEP(0u, 1u, 6u, 2u, true)     // slice=0 idx=1
    ARGON2D_LM3_STEP(1u, 2u, 5u, 4u, true)     // slice=1 idx=0
    ARGON2D_LM3_STEP(2u, 3u, 6u, 4u, true)     // slice=1 idx=1
    ARGON2D_LM3_STEP(3u, 4u, 5u, 6u, true)     // slice=2 idx=0
    ARGON2D_LM3_STEP(4u, 5u, 6u, 6u, true)     // slice=2 idx=1
    ARGON2D_LM3_STEP(5u, 6u, 5u, 0u, true)     // slice=3 idx=0
    ARGON2D_LM3_STEP(6u, 7u, 6u, 0u, true)     // slice=3 idx=1

    // Pass 2 — withXor = true (same pattern as pass 1)
    ARGON2D_LM3_STEP(7u, 0u, 5u, 2u, true)
    ARGON2D_LM3_STEP(0u, 1u, 6u, 2u, true)
    ARGON2D_LM3_STEP(1u, 2u, 5u, 4u, true)
    ARGON2D_LM3_STEP(2u, 3u, 6u, 4u, true)
    ARGON2D_LM3_STEP(3u, 4u, 5u, 6u, true)
    ARGON2D_LM3_STEP(4u, 5u, 6u, 6u, true)
    ARGON2D_LM3_STEP(5u, 6u, 5u, 0u, true)
    ARGON2D_LM3_STEP(6u, 7u, 6u, 0u, true)
}


///////////////////////////////////////////////////////////////////////////////
__host__
bool argon2d_lm3(
    cudaStream_t const stream,
    uint64_t* const    memory,
    uint32_t const     blocks,
    uint32_t const     threads)
{
    kernel_argon2d_lm3<<<blocks, threads, 0, stream>>>(memory);
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
