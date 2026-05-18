///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
#include <common/error/cuda_error.hpp>


///////////////////////////////////////////////////////////////////////////////
// Argon2d algorithm parameters — compile-time constants (p=1, m=8, t=3)
static constexpr uint32_t ARGON2D_LM1_T           { 3u   };
static constexpr uint32_t ARGON2D_LM1_Q           { 8u   };
static constexpr uint32_t ARGON2D_LM1_SEGMENT_LEN { 2u   };
static constexpr uint32_t ARGON2D_LM1_WORDS       { 128u };


///////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
uint64_t rotr64_lm1(uint64_t const x, uint32_t const n)
{
    return (x >> n) | (x << (64u - n));
}


///////////////////////////////////////////////////////////////////////////////
// Fills a 1024-byte Argon2d block with the seed value repeated across all words.
// Replaces the real Blake2b H' initialisation (RFC 9106 §3.2) with raw values
// so the benchmark measures only Argon2d work (G compression, fBlaMka,
// data-dependent memory traversal) without any Blake2b overhead.
__device__ __forceinline__
void fakeInit_lm1(uint64_t* const out, uint64_t const seed)
{
    for (uint32_t i{ 0u }; i < ARGON2D_LM1_WORDS; ++i)
    {
        out[i] = seed;
    }
}


///////////////////////////////////////////////////////////////////////////////
// Argon2 mixing function: fBlaMka replaces Blake2b's plain addition.
// a + b + 2*(a mod 2^32)*(b mod 2^32)  mod 2^64
__device__ __forceinline__
void G_mix_lm1(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d)
{
    a = a + b + 2ull * (a & 0xFFFFFFFFull) * (b & 0xFFFFFFFFull);
    d = rotr64_lm1(d ^ a, 32u);
    c = c + d + 2ull * (c & 0xFFFFFFFFull) * (d & 0xFFFFFFFFull);
    b = rotr64_lm1(b ^ c, 24u);
    a = a + b + 2ull * (a & 0xFFFFFFFFull) * (b & 0xFFFFFFFFull);
    d = rotr64_lm1(d ^ a, 16u);
    c = c + d + 2ull * (c & 0xFFFFFFFFull) * (d & 0xFFFFFFFFull);
    b = rotr64_lm1(b ^ c, 63u);
}


///////////////////////////////////////////////////////////////////////////////
// Blake2b-round permutation P on 16 words (BLAKE2_ROUND_NOMSG).
// Applies 4 column G_mix then 4 diagonal G_mix (RFC 9106 §3.4).
__device__ __forceinline__
void P_lm1(uint64_t* const v)
{
    G_mix_lm1(v[0],  v[4],  v[8],  v[12]);
    G_mix_lm1(v[1],  v[5],  v[9],  v[13]);
    G_mix_lm1(v[2],  v[6],  v[10], v[14]);
    G_mix_lm1(v[3],  v[7],  v[11], v[15]);

    G_mix_lm1(v[0],  v[5],  v[10], v[15]);
    G_mix_lm1(v[1],  v[6],  v[11], v[12]);
    G_mix_lm1(v[2],  v[7],  v[8],  v[13]);
    G_mix_lm1(v[3],  v[4],  v[9],  v[14]);
}


///////////////////////////////////////////////////////////////////////////////
// Argon2 block compression G(X, Y) -> Z  (RFC 9106 §3.4).
// The 128-word block is viewed as an 8x16 matrix.
//   Rows    : 8 groups of 16 consecutive words  (indices l*16 .. l*16+15)
//   Columns : 8 groups of 16 interleaved words
//             column l = indices 2*l, 2*l+1, 2*l+16, 2*l+17, ..., 2*l+112, 2*l+113
// withXor: true for passes > 0, XORs the new value with the old Z.
__device__
void G_block_lm1(
    uint64_t* const       Z,
    uint64_t const* const X,
    uint64_t const* const Y,
    bool const            withXor)
{
    uint64_t R[ARGON2D_LM1_WORDS];
    uint64_t Q[ARGON2D_LM1_WORDS];

    for (uint32_t i{ 0u }; i < ARGON2D_LM1_WORDS; ++i)
    {
        R[i] = X[i] ^ Y[i];
        Q[i] = R[i];
    }

    for (uint32_t l{ 0u }; l < 8u; ++l)
    {
        P_lm1(Q + l * 16u);
    }

    for (uint32_t l{ 0u }; l < 8u; ++l)
    {
        uint64_t v[16];

        for (uint32_t k{ 0u }; k < 8u; ++k)
        {
            v[k * 2u]      = Q[2u * l + k * 16u];
            v[k * 2u + 1u] = Q[2u * l + k * 16u + 1u];
        }

        P_lm1(v);

        for (uint32_t k{ 0u }; k < 8u; ++k)
        {
            Q[2u * l + k * 16u]      = v[k * 2u];
            Q[2u * l + k * 16u + 1u] = v[k * 2u + 1u];
        }
    }

    for (uint32_t i{ 0u }; i < ARGON2D_LM1_WORDS; ++i)
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
// Compute the reference column index (phi formula, RFC 9106 §3.3).
// For p=1 (single lane) same_lane is always true.
__device__ __forceinline__
uint32_t computeRefCol_lm1(
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
            refArea = static_cast<int32_t>(sliceN * ARGON2D_LM1_SEGMENT_LEN + idx) - 1;
        }
    }
    else
    {
        refArea = static_cast<int32_t>(ARGON2D_LM1_Q - ARGON2D_LM1_SEGMENT_LEN + idx) - 1;
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
        start = (sliceN + 1u) * ARGON2D_LM1_SEGMENT_LEN;
    }

    return static_cast<uint32_t>((static_cast<uint64_t>(start) + relativePos) % ARGON2D_LM1_Q);
}


///////////////////////////////////////////////////////////////////////////////
// 1 thread = 1 Argon2d solution, no optimisation.
// memory layout: threadId * Q * WORDS words per thread.
__global__
void kernel_argon2d_lm1(uint64_t* const memory)
{
    uint32_t const threadId{ blockIdx.x * blockDim.x + threadIdx.x };

    uint64_t* const B
    {
        memory + static_cast<uint64_t>(threadId) * ARGON2D_LM1_Q * ARGON2D_LM1_WORDS
    };

    fakeInit_lm1(B,                       static_cast<uint64_t>(threadId) * 2ull);
    fakeInit_lm1(B + ARGON2D_LM1_WORDS,   static_cast<uint64_t>(threadId) * 2ull + 1ull);

    for (uint32_t passN{ 0u }; passN < ARGON2D_LM1_T; ++passN)
    {
        for (uint32_t sliceN{ 0u }; sliceN < 4u; ++sliceN)
        {
            for (uint32_t idx{ 0u }; idx < ARGON2D_LM1_SEGMENT_LEN; ++idx)
            {
                if (0u == passN && 0u == sliceN && idx < 2u)
                {
                    continue;
                }

                uint32_t const col    { sliceN * ARGON2D_LM1_SEGMENT_LEN + idx };
                uint32_t const prevCol{ (col + ARGON2D_LM1_Q - 1u) % ARGON2D_LM1_Q };

                uint64_t const* const prevBlock{ B + prevCol * ARGON2D_LM1_WORDS };

                uint64_t const J  { prevBlock[0] };
                uint32_t const J1 { static_cast<uint32_t>(J & 0xFFFFFFFFull) };

                uint32_t const refCol{ computeRefCol_lm1(J1, passN, sliceN, idx) };

                G_block_lm1(
                    B + col    * ARGON2D_LM1_WORDS,
                    prevBlock,
                    B + refCol * ARGON2D_LM1_WORDS,
                    passN > 0u);
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////
__host__
bool argon2d_lm1(
    cudaStream_t const stream,
    uint64_t* const    memory,
    uint32_t const     blocks,
    uint32_t const     threads)
{
    kernel_argon2d_lm1<<<blocks, threads, 0, stream>>>(memory);
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
