///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
#include <common/cuda/rotate_byte.cuh>
#include <common/error/cuda_error.hpp>


__device__ __constant__
uint64_t BLAKE2B_LM3_IV[8]
{
    0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
    0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
    0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};


__device__ __constant__
uint8_t BLAKE2B_LM3_SIGMA[10][16]
{
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
};


// G as macro: operates directly on named variables
#define BLAKE2B_G_LM3(va, vb, vc, vd, x, y)     \
    {                                           \
        (va) = (va) + (vb) + (x);               \
        (vd) = ror_64((vd) ^ (va), 32u);        \
        (vc) = (vc) + (vd);                     \
        (vb) = ror_64((vb) ^ (vc), 24u);        \
        (va) = (va) + (vb) + (y);               \
        (vd) = ror_64((vd) ^ (va), 16u);        \
        (vc) = (vc) + (vd);                     \
        (vb) = ror_64((vb) ^ (vc), 63u);        \
    }


__global__
void kernel_blake2b_lm3()
{
    uint32_t const threadId{ blockIdx.x * blockDim.x + threadIdx.x };

    uint64_t m[16]{};
    m[0] = static_cast<uint64_t>(threadId);

    // h as explicit named registers
    uint64_t h0{ BLAKE2B_LM3_IV[0] ^ 0x0000000001010020ULL };
    uint64_t h1{ BLAKE2B_LM3_IV[1] };
    uint64_t h2{ BLAKE2B_LM3_IV[2] };
    uint64_t h3{ BLAKE2B_LM3_IV[3] };
    uint64_t h4{ BLAKE2B_LM3_IV[4] };
    uint64_t h5{ BLAKE2B_LM3_IV[5] };
    uint64_t h6{ BLAKE2B_LM3_IV[6] };
    uint64_t h7{ BLAKE2B_LM3_IV[7] };

    // Working vector v as explicit named registers
    uint64_t v0{ h0 };
    uint64_t v1{ h1 };
    uint64_t v2{ h2 };
    uint64_t v3{ h3 };
    uint64_t v4{ h4 };
    uint64_t v5{ h5 };
    uint64_t v6{ h6 };
    uint64_t v7{ h7 };
    uint64_t v8 { BLAKE2B_LM3_IV[0] };
    uint64_t v9 { BLAKE2B_LM3_IV[1] };
    uint64_t v10{ BLAKE2B_LM3_IV[2] };
    uint64_t v11{ BLAKE2B_LM3_IV[3] };
    uint64_t v12{ BLAKE2B_LM3_IV[4] ^ 8ULL };
    uint64_t v13{ BLAKE2B_LM3_IV[5] };
    uint64_t v14{ BLAKE2B_LM3_IV[6] ^ 0xFFFFFFFFFFFFFFFFULL };
    uint64_t v15{ BLAKE2B_LM3_IV[7] };

    #pragma unroll
    for (uint32_t r{ 0u }; r < 12u; ++r)
    {
        uint8_t const* const s{ BLAKE2B_LM3_SIGMA[r % 10u] };
        BLAKE2B_G_LM3(v0, v4, v8,  v12, m[s[0]],  m[s[1]]);
        BLAKE2B_G_LM3(v1, v5, v9,  v13, m[s[2]],  m[s[3]]);
        BLAKE2B_G_LM3(v2, v6, v10, v14, m[s[4]],  m[s[5]]);
        BLAKE2B_G_LM3(v3, v7, v11, v15, m[s[6]],  m[s[7]]);
        BLAKE2B_G_LM3(v0, v5, v10, v15, m[s[8]],  m[s[9]]);
        BLAKE2B_G_LM3(v1, v6, v11, v12, m[s[10]], m[s[11]]);
        BLAKE2B_G_LM3(v2, v7, v8,  v13, m[s[12]], m[s[13]]);
        BLAKE2B_G_LM3(v3, v4, v9,  v14, m[s[14]], m[s[15]]);
    }

    h0 ^= v0 ^ v8;
    h1 ^= v1 ^ v9;
    h2 ^= v2 ^ v10;
    h3 ^= v3 ^ v11;

}

#undef BLAKE2B_G_LM3


__host__
bool blake2b_lm3(
    cudaStream_t stream,
    uint32_t const blocks,
    uint32_t const threads)
{
    kernel_blake2b_lm3<<<blocks, threads, 0, stream>>>();
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
