///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
#include <common/error/cuda_error.hpp>


///////////////////////////////////////////////////////////////////////////////
static constexpr uint32_t ARGON2D_LM2_T           { 3u   };
static constexpr uint32_t ARGON2D_LM2_Q           { 8u   };
static constexpr uint32_t ARGON2D_LM2_SEGMENT_LEN { 2u   };
static constexpr uint32_t ARGON2D_LM2_WORDS       { 128u };


///////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
uint64_t rotr64_lm2(uint64_t const x, uint32_t const n)
{
    return (x >> n) | (x << (64u - n));
}


///////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void fakeInit_lm2(uint64_t* const out, uint64_t const seed)
{
    #pragma unroll
    for (uint32_t i{ 0u }; i < ARGON2D_LM2_WORDS; ++i)
    {
        out[i] = seed;
    }
}


///////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void G_mix_lm2(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d)
{
    a = a + b + 2ull * (a & 0xFFFFFFFFull) * (b & 0xFFFFFFFFull);
    d = rotr64_lm2(d ^ a, 32u);
    c = c + d + 2ull * (c & 0xFFFFFFFFull) * (d & 0xFFFFFFFFull);
    b = rotr64_lm2(b ^ c, 24u);
    a = a + b + 2ull * (a & 0xFFFFFFFFull) * (b & 0xFFFFFFFFull);
    d = rotr64_lm2(d ^ a, 16u);
    c = c + d + 2ull * (c & 0xFFFFFFFFull) * (d & 0xFFFFFFFFull);
    b = rotr64_lm2(b ^ c, 63u);
}


///////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
void P_lm2(uint64_t* const v)
{
    G_mix_lm2(v[0],  v[4],  v[8],  v[12]);
    G_mix_lm2(v[1],  v[5],  v[9],  v[13]);
    G_mix_lm2(v[2],  v[6],  v[10], v[14]);
    G_mix_lm2(v[3],  v[7],  v[11], v[15]);

    G_mix_lm2(v[0],  v[5],  v[10], v[15]);
    G_mix_lm2(v[1],  v[6],  v[11], v[12]);
    G_mix_lm2(v[2],  v[7],  v[8],  v[13]);
    G_mix_lm2(v[3],  v[4],  v[9],  v[14]);
}


///////////////////////////////////////////////////////////////////////////////
__device__
void G_block_lm2(
    uint64_t* const       Z,
    uint64_t const* const X,
    uint64_t const* const Y,
    bool const            withXor)
{
    uint64_t R[ARGON2D_LM2_WORDS];
    uint64_t Q[ARGON2D_LM2_WORDS];

    #pragma unroll
    for (uint32_t i{ 0u }; i < ARGON2D_LM2_WORDS; ++i)
    {
        R[i] = X[i] ^ Y[i];
        Q[i] = R[i];
    }

    #pragma unroll
    for (uint32_t l{ 0u }; l < 8u; ++l)
    {
        P_lm2(Q + l * 16u);
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

        P_lm2(v);

        #pragma unroll
        for (uint32_t k{ 0u }; k < 8u; ++k)
        {
            Q[2u * l + k * 16u]      = v[k * 2u];
            Q[2u * l + k * 16u + 1u] = v[k * 2u + 1u];
        }
    }

    #pragma unroll
    for (uint32_t i{ 0u }; i < ARGON2D_LM2_WORDS; ++i)
    {
        uint64_t z{ Q[i] ^ R[i] };
        if (withXor)
        {
            z ^= Z[i];
        }
        Z[i] = z;
    }
}


///////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
uint32_t computeRefCol_lm2(
    uint64_t const J1,
    uint32_t const passN,
    uint32_t const sliceN,
    uint32_t const idx)
{
    int32_t refArea;

    if (0u == passN)
    {
        if (0u == sliceN)
        {
            refArea = static_cast<int32_t>(idx) - 1;
        }
        else
        {
            refArea = static_cast<int32_t>(sliceN * ARGON2D_LM2_SEGMENT_LEN + idx) - 1;
        }
    }
    else
    {
        refArea = static_cast<int32_t>(ARGON2D_LM2_Q - ARGON2D_LM2_SEGMENT_LEN + idx) - 1;
    }

    if (refArea < 1)
    {
        refArea = 1;
    }

    uint64_t const x          { J1 };
    uint64_t const stepA      { (x * x) >> 32u };
    uint64_t const stepB      { (static_cast<uint64_t>(refArea) * stepA) >> 32u };
    uint64_t const relativePos{ static_cast<uint64_t>(refArea) - 1ull - stepB };

    uint32_t start;
    if (0u == passN)
    {
        start = 0u;
    }
    else if (3u == sliceN)
    {
        start = 0u;
    }
    else
    {
        start = (sliceN + 1u) * ARGON2D_LM2_SEGMENT_LEN;
    }

    return static_cast<uint32_t>((static_cast<uint64_t>(start) + relativePos) % ARGON2D_LM2_Q);
}


///////////////////////////////////////////////////////////////////////////////
__global__
void kernel_argon2d_lm2(uint64_t* const memory)
{
    uint32_t const threadId{ blockIdx.x * blockDim.x + threadIdx.x };

    uint64_t* const B
    {
        memory + static_cast<uint64_t>(threadId) * ARGON2D_LM2_Q * ARGON2D_LM2_WORDS
    };

    fakeInit_lm2(B,                       static_cast<uint64_t>(threadId) * 2ull);
    fakeInit_lm2(B + ARGON2D_LM2_WORDS,   static_cast<uint64_t>(threadId) * 2ull + 1ull);

    for (uint32_t passN{ 0u }; passN < ARGON2D_LM2_T; ++passN)
    {
        for (uint32_t sliceN{ 0u }; sliceN < 4u; ++sliceN)
        {
            for (uint32_t idx{ 0u }; idx < ARGON2D_LM2_SEGMENT_LEN; ++idx)
            {
                if (0u == passN && 0u == sliceN && idx < 2u)
                {
                    continue;
                }

                uint32_t const col    { sliceN * ARGON2D_LM2_SEGMENT_LEN + idx };
                uint32_t const prevCol{ (col + ARGON2D_LM2_Q - 1u) % ARGON2D_LM2_Q };

                uint64_t const* const prevBlock{ B + prevCol * ARGON2D_LM2_WORDS };

                uint64_t const J  { prevBlock[0] };
                uint32_t const J1 { static_cast<uint32_t>(J & 0xFFFFFFFFull) };

                uint32_t const refCol{ computeRefCol_lm2(J1, passN, sliceN, idx) };

                G_block_lm2(
                    B + col    * ARGON2D_LM2_WORDS,
                    prevBlock,
                    B + refCol * ARGON2D_LM2_WORDS,
                    passN > 0u);
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////
__host__
bool argon2d_lm2(
    cudaStream_t const stream,
    uint64_t* const    memory,
    uint32_t const     blocks,
    uint32_t const     threads)
{
    kernel_argon2d_lm2<<<blocks, threads, 0, stream>>>(memory);
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
