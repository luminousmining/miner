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
    #pragma unroll
    for (uint64_t i = 0ull; i < algo::LEN_HASH_512_WORD_32; ++i)
    {
        item[i] = seed_cache[i];
        light_cache[i] = seed_cache[i];
    }

    ///////////////////////////////////////////////////////////////////////////
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

    ///////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (uint64_t round = 0ull; round < algo::ethash::LIGHT_CACHE_ROUNDS; ++round)
    {
        uint32_t const number_item_u32 = castU32(light_cache_number);
        for (uint64_t i = 0ull; i < light_cache_number; ++i)
        {
            // arr = uint32_t[]
            // length = len(arr)
            // hash256_word32 = sizeof(has256::workd32) = 16
            // 
            // A = arr[(hash256_word32 * i) % length]
            // B arr[(length + (i -1)) % length]
            // C = A ^ B
            // C = keccak(C)
            // arr[hash256_word32 * i] = C
            //
            // arr[hash256_word32 * i] = keccak(arr[(hash256_word32 * i) % length] ^ arr[(length + (i -1)) % length])
            //
            // arr[16 * i] = keccak(arr[(16 * i) % length] ^ arr[(length + (i -1)) % length])

            uint32_t const start_index = algo::LEN_HASH_512_WORD_32 * i;
            uint32_t const fi = light_cache[start_index] % number_item_u32;
            uint32_t const si = (number_item_u32 + (castU32(i) - 1u)) % number_item_u32;

            uint32_t hash_xored[algo::LEN_HASH_512_WORD_32];
            xor_buffer<uint32_t, (uint32_t)algo::LEN_HASH_512_WORD_32>
            (
                hash_xored,
                light_cache + (fi * algo::LEN_HASH_512_WORD_32),
                light_cache + (si * algo::LEN_HASH_512_WORD_32)
            );

            keccak_f1600_u32(hash_xored);

            #pragma unroll
            for (uint32_t j = 0u; j < algo::LEN_HASH_512_WORD_32; ++j)
            {
                light_cache[start_index + j] = hash_xored[j];
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
