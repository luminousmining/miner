#pragma once

#include <algo/hash.hpp>
#include <algo/dag_context.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>


__host__
bool ethashFreeMemory(
    resolver::nvidia::ethash::KernelParameters& params)
{
    CU_SAFE_DELETE(params.lightCache);
    CU_SAFE_DELETE(params.seedCache);
    CU_SAFE_DELETE(params.dagCache);
    CU_SAFE_DELETE_HOST(params.resultCache);
    return true;
}


__host__
bool ethashInitMemory(
    algo::DagContext const& context,
    resolver::nvidia::ethash::KernelParameters& params,
    bool const buildLightCacheOnGPU)
{
    ////////////////////////////////////////////////////////////////////////////
    if (false == ethashFreeMemory(params))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    CU_ALLOC(&params.seedCache, algo::LEN_HASH_512);
    CU_ALLOC(&params.lightCache, context.lightCache.size);
    CU_ALLOC(&params.dagCache, context.dagCache.size);
    CU_ALLOC_HOST(&params.resultCache, sizeof(algo::ethash::Result) * 2u);

    ////////////////////////////////////////////////////////////////////////////
    params.resultCache[0].found = false;
    params.resultCache[0].count = 0u;
    params.resultCache[1].found = false;
    params.resultCache[1].count = 0u;
    for (uint32_t i{ 0u }; i < algo::ethash::MAX_RESULT; ++i)
    {
        params.resultCache[0].nonces[i] = 0ull;
        params.resultCache[1].nonces[i] = 0ull;
    }

    ////////////////////////////////////////////////////////////////////////////
    IS_NULL(params.lightCache);
    IS_NULL(params.dagCache);
    IS_NULL(params.seedCache);
    IS_NULL(params.resultCache);

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
    uint32_t const dagNumberItemU32{ castU32(context.dagCache.numberItem) };
    CUDA_ER(
        cudaMemcpyToSymbol(
            d_dag_number_item,
            (void*)&dagNumberItemU32,
            sizeof(uint32_t)));

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const lightNnumberItem{ castU32(context.lightCache.numberItem) };
    CUDA_ER(
        cudaMemcpyToSymbol(
            d_light_number_item,
            (void*)&lightNnumberItem,
            sizeof(uint32_t)));

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


__host__
bool ethashUpdateConstants(
    uint32_t const* const header,
    uint64_t const boundary)
{
    uint4 const* const headerU4{ (uint4 const*)header };
    CUDA_ER(cudaMemcpyToSymbol(d_header, headerU4, sizeof(uint4) * 2));
    CUDA_ER(cudaMemcpyToSymbol(d_boundary, (void*)&boundary, sizeof(uint64_t)));

    return true;
}
