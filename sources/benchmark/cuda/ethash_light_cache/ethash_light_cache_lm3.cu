///////////////////////////////////////////////////////////////////////////////
#include "benchmark/cuda/common/common.cuh"
#include "algo/crypto/cuda/keccak_f1600.cuh"

///////////////////////////////////////////////////////////////////////////////
#include "common/cast.hpp"
#include <algo/ethash/ethash.hpp>


__global__
void kernel_ethash_light_cache_initialize_seed_lm3(
    uint32_t* __restrict__ const light_cache,
    uint32_t const* __restrict__ const seed_cache)
{
    /////////////////////////////////////////////////////////////////////////
    uint32_t const thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    light_cache[thread_id] = seed_cache[thread_id];
}


__global__
void kernel_ethash_light_cache_initialize_keecak_lm3(
    uint32_t* __restrict__ const light_cache,
    uint32_t const* __restrict__ const seed_cache,
    uint64_t const light_cache_number)
{
    /////////////////////////////////////////////////////////////////////////
    uint32_t item[algo::LEN_HASH_512_WORD_32];

    /////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (uint64_t i = 0ull; i < algo::LEN_HASH_512_WORD_32; ++i)
    {
        item[i] = seed_cache[i];
    }

    /////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (uint64_t i = 1ull; i < light_cache_number; ++i)
    {
        keccak_f1600_u32(item);
        uint32_t const start_index = (i * algo::LEN_HASH_512_WORD_32);
        for (uint32_t j = 0u; j < algo::LEN_HASH_512_WORD_32; ++j)
        {
            light_cache[start_index + j] = item[j];
        }
    }
}


__global__
void kernel_ethash_light_cache_xor_lm3(
    uint32_t* const light_cache,
    uint32_t const light_cache_number_u32,
    uint64_t const light_cache_number)
{
    ///////////////////////////////////////////////////////////////////////////
    __shared__ uint32_t hash_xored[algo::LEN_HASH_512_WORD_32];
    uint32_t const thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

    ///////////////////////////////////////////////////////////////////////////
    for (uint64_t i = 0ull; i < light_cache_number; ++i)
    {
        ///////////////////////////////////////////////////////////////////////
        uint32_t const start_index = algo::LEN_HASH_512_WORD_32 * i;

        ///////////////////////////////////////////////////////////////////////
        uint32_t index_first = 0u;
        if (thread_id == 0u)
        {
            index_first = light_cache[start_index] % light_cache_number_u32;
            index_first *= algo::LEN_HASH_512_WORD_32;
        }
        index_first = reg_load(index_first, 0, algo::LEN_HASH_512_WORD_32);
        index_first += thread_id;

        ///////////////////////////////////////////////////////////////////////
        uint32_t index_second = (light_cache_number_u32 + (castU32(i) - 1u)) % light_cache_number_u32;
        index_second *= algo::LEN_HASH_512_WORD_32;
        index_second += thread_id;

        ///////////////////////////////////////////////////////////////////////
        uint32_t const a = light_cache[index_first];
        uint32_t const b = light_cache[index_second];
        uint32_t const c = a ^ b;
        hash_xored[thread_id] = c;
        __syncthreads();

        ///////////////////////////////////////////////////////////////////////
        if (thread_id == 0u)
        {
            keccak_f1600_u32(hash_xored);
        }

        ///////////////////////////////////////////////////////////////////////
        light_cache[start_index + thread_id] = hash_xored[thread_id];
    }
}


__host__
bool etash_light_cache_lm3(
    cudaStream_t stream,
    uint32_t* lightCache,
    uint32_t const* const seedCache,
    uint64_t const lightCacheNumber)
{
    ///////////////////////////////////////////////////////////////////////////
    kernel_ethash_light_cache_initialize_seed_lm3<<<1, algo::LEN_HASH_512_WORD_32, 0, stream>>>
    (
        lightCache,
        seedCache
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    ///////////////////////////////////////////////////////////////////////////
    kernel_ethash_light_cache_initialize_keecak_lm3<<<1, 1, 0, stream>>>
    (
        lightCache,
        seedCache,
        lightCacheNumber
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    ///////////////////////////////////////////////////////////////////////////
    for (uint64_t round = 0ull; round < algo::ethash::LIGHT_CACHE_ROUNDS; ++round)
    {
        kernel_ethash_light_cache_xor_lm3<<<1, algo::LEN_HASH_512_WORD_32, 0, stream>>>
        (
            lightCache,
            castU32(lightCacheNumber),
            lightCacheNumber
        );
        CUDA_ER(cudaStreamSynchronize(stream));
        CUDA_ER(cudaGetLastError());
    }

    ///////////////////////////////////////////////////////////////////////////
    return true;
}
