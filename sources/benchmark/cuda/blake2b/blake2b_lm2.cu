///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
#include <common/cuda/rotate_byte.cuh>
#include <common/error/cuda_error.hpp>


__device__ __constant__
uint64_t BLAKE2B_LM2_IV[8]
{
    0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
    0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
    0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};


__device__ __constant__
uint8_t BLAKE2B_LM2_SIGMA[10][16]
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


__device__ __forceinline__
void blake2b_G_lm2(
    uint64_t& a,
    uint64_t& b,
    uint64_t& c,
    uint64_t& d,
    uint64_t const x,
    uint64_t const y)
{
    a = a + b + x;
    d = ror_64(d ^ a, 32u);
    c = c + d;
    b = ror_64(b ^ c, 24u);
    a = a + b + y;
    d = ror_64(d ^ a, 16u);
    c = c + d;
    b = ror_64(b ^ c, 63u);
}


__device__ __forceinline__
void blake2b_compress_lm2(
    uint64_t h[8],
    uint64_t const m[16],
    uint64_t const t,
    uint32_t const last)
{
    uint64_t v[16];

    #pragma unroll
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        v[i]     = h[i];
        v[i + 8] = BLAKE2B_LM2_IV[i];
    }

    v[12] ^= t;

    if (0u != last)
    {
        v[14] ^= 0xFFFFFFFFFFFFFFFFULL;
    }

    #pragma unroll
    for (uint32_t r{ 0u }; r < 12u; ++r)
    {
        uint8_t const* const s{ BLAKE2B_LM2_SIGMA[r % 10u] };
        blake2b_G_lm2(v[0], v[4], v[8],  v[12], m[s[0]],  m[s[1]]);
        blake2b_G_lm2(v[1], v[5], v[9],  v[13], m[s[2]],  m[s[3]]);
        blake2b_G_lm2(v[2], v[6], v[10], v[14], m[s[4]],  m[s[5]]);
        blake2b_G_lm2(v[3], v[7], v[11], v[15], m[s[6]],  m[s[7]]);
        blake2b_G_lm2(v[0], v[5], v[10], v[15], m[s[8]],  m[s[9]]);
        blake2b_G_lm2(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        blake2b_G_lm2(v[2], v[7], v[8],  v[13], m[s[12]], m[s[13]]);
        blake2b_G_lm2(v[3], v[4], v[9],  v[14], m[s[14]], m[s[15]]);
    }

    #pragma unroll
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        h[i] ^= v[i] ^ v[i + 8u];
    }
}


__global__
void kernel_blake2b_lm2()
{
    uint32_t const threadId{ blockIdx.x * blockDim.x + threadIdx.x };

    uint64_t h[8];

    #pragma unroll
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        h[i] = BLAKE2B_LM2_IV[i];
    }

    // Parameter block : digest_size=32, key_len=0, fanout=1, depth=1
    h[0] ^= 0x0000000001010020ULL;

    uint64_t m[16]{};
    m[0] = static_cast<uint64_t>(threadId);

    blake2b_compress_lm2(h, m, 8ull, 1u);

}


__host__
bool blake2b_lm2(
    cudaStream_t stream,
    uint32_t const blocks,
    uint32_t const threads)
{
    kernel_blake2b_lm2<<<blocks, threads, 0, stream>>>();
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
