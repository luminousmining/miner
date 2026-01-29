///////////////////////////////////////////////////////////////////////////////
#include <benchmark/cuda/common/common.cuh>
#include <algo/crypto/cuda/keccak_f1600.cuh>

///////////////////////////////////////////////////////////////////////////////
#include <common/cast.hpp>
#include <algo/ethash/ethash.hpp>


__global__
void kernel_etash_light_cache_lm1(
    uint32_t* __restrict__ light_cache,
    uint32_t const* __restrict__ const seed_cache,
    uint64_t const light_cache_number)
{
    /////////////////////////////////////////////////////////////////////////
    uint32_t item[algo::LEN_HASH_512_WORD_32];

    /////////////////////////////////////////////////////////////////////////
    for (uint64_t i = 0ull; i < algo::LEN_HASH_512_WORD_32; ++i)
    {
        item[i] = seed_cache[i];
        light_cache[i] = seed_cache[i];
    }

    ///////////////////////////////////////////////////////////////////////////
    for (uint64_t i = 1ull; i < light_cache_number; ++i)
    {
        keccak_f1600_u32(item);
        uint32_t const start_index = (i * algo::LEN_HASH_512_WORD_32);
        for (uint32_t j = 0u; j < algo::LEN_HASH_512_WORD_32; ++j)
        {
            light_cache[start_index + j] = item[j];
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    for (uint64_t round = 0ull; round < algo::ethash::LIGHT_CACHE_ROUNDS; ++round)
    {
        uint32_t const number_item_u32 = castU32(light_cache_number);
        for (uint64_t i = 0ull; i < light_cache_number; ++i)
        {
            uint32_t const start_index = algo::LEN_HASH_512_WORD_32 * i;
            uint32_t const fi = light_cache[start_index] % number_item_u32;
            uint32_t const si = (number_item_u32 + (castU32(i) - 1u)) % number_item_u32;

            uint32_t hashXored[32];
            xor_buffer<uint32_t, (uint32_t)algo::LEN_HASH_512_WORD_32>
            (
                hashXored,
                light_cache + (fi * algo::LEN_HASH_512_WORD_32),
                light_cache + (si * algo::LEN_HASH_512_WORD_32)
            );

            keccak_f1600_u32(hashXored);

            for (uint32_t j = 0u; j < algo::LEN_HASH_512_WORD_32; ++j)
            {
                light_cache[start_index + j] = hashXored[j];
            }
        }
    }
}


__host__
bool etash_light_cache_lm1(
    cudaStream_t stream,
    uint32_t* lightCache,
    uint32_t const* const seedCache,
    uint64_t const lightCacheNumber)
{
    kernel_etash_light_cache_lm1<<<1, 1, 0, stream>>>
    (
        lightCache,
        seedCache,
        lightCacheNumber
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());
    return true;
}
