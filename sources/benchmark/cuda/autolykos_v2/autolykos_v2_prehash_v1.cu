///////////////////////////////////////////////////////////////////////////////
#include <benchmark/cuda/kernels.hpp>

///////////////////////////////////////////////////////////////////////////////
#include <benchmark/cuda/common/common.cuh>
#include <algo/crypto/cuda/blake2b.cuh>


__global__
void kernel_autolykos_v2_prehash_lm1(
    uint32_t* const __restrict__ hashes,
    uint32_t const period,
    uint32_t const height)
{
    ///////////////////////////////////////////////////////////////////////////
    uint32_t const tid{ threadIdx.x + (blockDim.x * blockIdx.x) };
    if (tid >= period)
    {
        return;
    }

    ///////////////////////////////////////////////////////////////////////////
    uint64_t h[8]
    {
        0x6A09E667F3BCC908UL,
        0xBB67AE8584CAA73BUL,
        0x3C6EF372FE94F82BUL,
        0xA54FF53A5F1D36F1UL,
        0x510E527FADE682D1UL,
        0x9B05688C2B3E6C1FUL,
        0x1F83D9ABFB41BD6BUL,
        0x5BE0CD19137E2179UL
    };
    uint64_t b[16];
    uint64_t t = 0ull;

    h[0] ^= 0x01010020;
    b[0] = 0ull;

    ///////////////////////////////////////////////////////////////////////////
    uint64_t ctr = 0;
    ((uint32_t*)b)[0] = be_u32(tid);
    ((uint32_t*)b)[1] = height;
    for (uint32_t x = 1u; x < 16u; ++x, ++ctr)
    {
        b[x] = be_u64(ctr);
    }

    ///////////////////////////////////////////////////////////////////////////
    #pragma unroll 1
    for (uint32_t z = 0; z < 63u; ++z)
    {
        t += 128ull;
        blake2b((uint64_t*)h, (uint64_t *)b, t, 0UL);

        #pragma unroll
        for (uint32_t x = 0; x < 16u; ++x, ++ctr)
        {
            b[x] = be_u64(ctr);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    t += 128ull;
    blake2b((uint64_t*)h, (uint64_t*)b, t, 0UL);
    b[0] = be_u64(ctr);
    t += 8;

    ///////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (uint32_t i = 1u; i < 16u; ++i)
    {
        ((uint64_t*)b)[i] = 0ull;
    }

    ///////////////////////////////////////////////////////////////////////////
    blake2b((uint64_t*)h, (uint64_t*)b, t, 0xFFFFFFFFFFFFFFFFUL);

    ///////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (uint32_t i = 0u; i < 4u; ++i)
    {
        ((uint64_t*)hashes)[(tid + 1) * 4 - i - 1] = be_u64(h[i]);
    }

    ///////////////////////////////////////////////////////////////////////////
    ((uint8_t*)hashes)[tid * 32 + 31] = 0;
}


__host__
bool autolykos_v2_prehash_lm1(
    cudaStream_t stream,
    uint32_t* dag,
    uint32_t const blocks,
    uint32_t const threads,
    uint32_t const period,
    uint32_t const height)
{
    kernel_autolykos_v2_prehash_lm1<<<blocks, threads, 0, stream>>>
    (
        dag,
        period,
        height
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
