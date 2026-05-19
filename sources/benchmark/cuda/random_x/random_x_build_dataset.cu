///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
#include <common/error/cuda_error.hpp>


///////////////////////////////////////////////////////////////////////////////
constexpr uint64_t RANDOMX_DATASET_ITEMS{ 34078720ull }; // (2^31 + 32 MiB) / 64
constexpr uint64_t RANDOMX_CACHE_ITEMS  { 4194304ull };  // 256 MiB / 64
constexpr uint64_t DATASET_LCG_MUL      { 6364136223846793005ULL };
constexpr uint64_t DATASET_LCG_ADD      { 1442695040888963407ULL };


// Each GPU thread fills one 64-byte dataset item from the 256 MiB cache.
// Uses LCG initialization + 8 cache reads to simulate SuperscalarHash.
__global__
void kernel_random_x_build_dataset(
    uint64_t const* const cache,
    uint64_t* const       dataset)
{
    uint64_t const stride{ static_cast<uint64_t>(gridDim.x) * blockDim.x };
    uint64_t       item  { static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x };

    while (item < RANDOMX_DATASET_ITEMS)
    {
        // Initialize r[0..7] with LCG seeded from item index
        uint64_t r[8];
        uint64_t lcg{ item };
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            lcg  = lcg * DATASET_LCG_MUL + DATASET_LCG_ADD;
            r[i] = lcg;
        }

        // 8 cache reads with XOR mixing (simplified SuperscalarHash)
        for (uint32_t j{ 0u }; j < 8u; ++j)
        {
            uint64_t const        cacheIdx{ r[j % 8u] % RANDOMX_CACHE_ITEMS };
            uint64_t const* const cblk    { cache + cacheIdx * 8ull };

            for (uint32_t i{ 0u }; i < 8u; ++i)
            {
                r[i] ^= cblk[i];
            }

            r[0] = r[0] * DATASET_LCG_MUL + DATASET_LCG_ADD;
        }

        uint64_t* const dest{ dataset + item * 8ull };
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            dest[i] = r[i];
        }

        item += stride;
    }
}


__host__
bool random_x_build_dataset(
    cudaStream_t         stream,
    uint8_t const* const cache,
    uint64_t* const      dataset)
{
    constexpr uint32_t BUILD_BLOCKS { 4096u };
    constexpr uint32_t BUILD_THREADS{ 1024u };

    kernel_random_x_build_dataset<<<BUILD_BLOCKS, BUILD_THREADS, 0, stream>>>(
        reinterpret_cast<uint64_t const*>(cache),
        dataset);
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
