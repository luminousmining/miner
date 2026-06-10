#include "kernel/common/rotate_byte.cl"
#include "kernel/common/xor.cl"

#include "kernel/crypto/keccak_f1600.cl"


__kernel
void ethash_light_cache_lm0(
    __global uint4* const restrict cache,
    uint const cache_number_item)
{
    ////////////////////////////////////////////////////////////////////////////
    // The host wrote the 64 bytes hashed seed into cache[0..3].
    uint4 item[4];
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 4u; ++i)
    {
        item[i] = cache[i];
    }

    ////////////////////////////////////////////////////////////////////////////
    for (uint i = 1u; i < cache_number_item; ++i)
    {
        keccak_f1600(item);
        uint const start_index = i * 4u;
        __attribute__((opencl_unroll_hint))
        for (uint j = 0u; j < 4u; ++j)
        {
            cache[start_index + j] = item[j];
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    for (uint round = 0u; round < LIGHT_CACHE_ROUNDS; ++round)
    {
        for (uint i = 0u; i < cache_number_item; ++i)
        {
            uint const start_index = i * 4u;
            uint const fi = (cache[start_index].x % cache_number_item) * 4u;
            uint const si = ((cache_number_item + (i - 1u)) % cache_number_item) * 4u;

            uint4 xored[4];
            __attribute__((opencl_unroll_hint))
            for (uint j = 0u; j < 4u; ++j)
            {
                xored[j] = cache[fi + j] ^ cache[si + j];
            }

            keccak_f1600(xored);

            __attribute__((opencl_unroll_hint))
            for (uint j = 0u; j < 4u; ++j)
            {
                cache[start_index + j] = xored[j];
            }
        }
    }
}
