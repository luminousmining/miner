#pragma once

#include <common/cast.hpp>
#include <common/cuda/debug.cuh>
#include <common/cuda/xor.cuh>
#include <algo/ethash/ethash.hpp>
#include <algo/crypto/cuda/keccak_f1600.cuh>


__global__
void kernel_progpow_build_light_cache(
    uint32_t const* const seed)
{
    /////////////////////////////////////////////////////////////////////////
    uint32_t* light_cache = (uint32_t*)d_light_cache;
    uint32_t item[algo::LEN_HASH_512_WORD_32];

    /////////////////////////////////////////////////////////////////////////
    for (uint64_t i = 0ull; i < algo::LEN_HASH_512_WORD_32; ++i)
    {
        item[i] = seed[i];
        light_cache[i] = seed[i];
    }

    ///////////////////////////////////////////////////////////////////////////
    for (uint64_t i = 1ull; i < d_light_number_item; ++i)
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
        uint32_t const number_item_u32 = castU32(d_light_number_item);
        for (uint64_t i = 0ull; i < d_light_number_item; ++i)
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
bool progpowBuildLightCache(
    cudaStream_t stream,
    uint32_t const* const seed)
{
    kernel_progpow_build_light_cache<<<1, 1, 0, stream>>>(seed);
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
