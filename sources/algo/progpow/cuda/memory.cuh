#pragma once

#include <algo/hash.hpp>
#include <algo/dag_context.hpp>
#include <common/error/cuda_error.hpp>
#include <common/custom.hpp>


__host__
bool progpowFreeMemory(
    resolver::nvidia::progpow::KernelParameters& params)
{
    CU_SAFE_DELETE(params.lightCache);
    CU_SAFE_DELETE(params.seedCache);
    CU_SAFE_DELETE(params.dagCache);
    CU_SAFE_DELETE(params.headerCache);
    CU_SAFE_DELETE_HOST(params.resultCache);

    return true;
}


__host__
bool progpowInitMemory(
    algo::DagContext const& context,
    resolver::nvidia::progpow::KernelParameters& params,
    bool const buildLightCacheOnGPU)
{
    ////////////////////////////////////////////////////////////////////////////
    if (false == progpowFreeMemory(params))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == buildLightCacheOnGPU)
    {
        CU_ALLOC(&params.seedCache, algo::LEN_HASH_512);
    }

    ////////////////////////////////////////////////////////////////////////////
    CU_ALLOC(&params.lightCache, context.lightCache.size);
    CU_ALLOC(&params.dagCache, context.dagCache.size);
    CU_ALLOC(&params.headerCache, sizeof(uint32_t) * algo::LEN_HASH_256_WORD_32);
    CU_ALLOC_HOST(&params.resultCache, sizeof(algo::progpow::Result) * 2u);

    ////////////////////////////////////////////////////////////////////////////
    if (true == buildLightCacheOnGPU)
    {
        IS_NULL(params.seedCache);
    }

    ////////////////////////////////////////////////////////////////////////////
    IS_NULL(params.lightCache);
    IS_NULL(params.dagCache);
    IS_NULL(params.headerCache);
    IS_NULL(params.resultCache);

    ////////////////////////////////////////////////////////////////////////////
    params.resultCache[0].count = 0u;
    params.resultCache[0].found = false;
    params.resultCache[1].count = 0u;
    params.resultCache[1].found = false;
    for (uint32_t i{ 0u }; i < algo::progpow::MAX_RESULT; ++i)
    {
        params.resultCache[0].nonces[i] = 0ull;
        params.resultCache[1].nonces[i] = 0ull;
        for (uint32_t x{ 0u }; x < algo::LEN_HASH_256_WORD_32; ++x)
        {
            params.resultCache[0].hash[i][x] = 0u;
            params.resultCache[1].hash[i][x] = 0u;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == buildLightCacheOnGPU)
    {
        CUDA_ER(
            cudaMemcpy(
                params.seedCache,
                context.hashedSeedCache.word32,
                algo::LEN_HASH_512_WORD_32 * sizeof(uint32_t),
                cudaMemcpyHostToDevice));
    }
    else
    {
        CUDA_ER(
            cudaMemcpy(
                (void*)params.lightCache,
                (void const*)context.lightCache.hash->bytes,
                context.lightCache.size,
                cudaMemcpyHostToDevice));
    }

    ////////////////////////////////////////////////////////////////////////////
    CUDA_ER(
        cudaMemcpyToSymbol(
            d_light_cache,
            (void**)&params.lightCache,
            sizeof(uint4*)));
    CUDA_ER(
        cudaMemcpyToSymbol(
            d_dag,
            (void**)&params.dagCache,
            sizeof(uint4*)));

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const dagNumberItemU32{ (uint32_t)context.dagCache.numberItem };
    CUDA_ER(
        cudaMemcpyToSymbol(
            d_dag_number_item,
            (void*)&dagNumberItemU32,
            sizeof(uint32_t)));

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const lightNnumberItem{ (uint32_t)context.lightCache.numberItem };
    CUDA_ER(
        cudaMemcpyToSymbol(
            d_light_number_item,
            (void*)&lightNnumberItem,
            sizeof(uint32_t)));

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


__host__
bool progpowUpdateConstants(
    uint32_t const* const hostBuffer,
    uint32_t* const deviceBuffer)
{
    CUDA_ER(
        cudaMemcpy(
            deviceBuffer,
            hostBuffer,
            algo::LEN_HASH_256_WORD_32 * sizeof(uint32_t),
            cudaMemcpyHostToDevice));

    return true;
}
