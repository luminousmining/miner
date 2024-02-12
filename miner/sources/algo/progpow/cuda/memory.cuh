#pragma once

#include <algo/hash.hpp>

#include <common/custom.hpp>


__host__
bool progpowInitMemory(
    algo::DagContext const& context,
    resolver::nvidia::progpow::KernelParameters& params)
{
    CU_SAFE_DELETE(params.lightCache);
    CU_SAFE_DELETE(params.dagCache);
    CU_SAFE_DELETE(params.headerCache);
    CU_SAFE_DELETE_HOST(params.resultCache);

    CUDA_ER(cudaMalloc((void**)&params.lightCache, context.lightCache.size));
    CUDA_ER(cudaMalloc((void**)&params.dagCache, context.dagCache.size));
    CUDA_ER(cudaMalloc((void**)&params.headerCache, sizeof(uint32_t) * algo::LEN_HASH_256_WORD_32));

    CUDA_ER(cudaMemcpy((void*)params.lightCache,
                       (void const*)context.lightCache.hash->bytes,
                       context.lightCache.size,
                       cudaMemcpyHostToDevice));

    CUDA_ER(cudaMallocHost((void**)&params.resultCache, sizeof(algo::progpow::Result), 0));
    params.resultCache->count = 0u;
    params.resultCache->found = false;

    for (uint32_t i{ 0u }; i < 4u; ++i)
    {
        params.resultCache->nonces[i] = 0ull;
    }
    for (uint32_t i{ 0u }; i < 4u; ++i)
    {
        for (uint32_t x{ 0u }; x < 8u; ++x)
        {
            params.resultCache->hash[i][x] = 0u;
        }
    }

    CUDA_ER(cudaMemcpyToSymbol(d_light_cache, (void**)&params.lightCache, sizeof(uint4*)));
    CUDA_ER(cudaMemcpyToSymbol(d_dag, (void**)&params.dagCache, sizeof(uint4*)));

    uint32_t const dagNumberItemU32{ (uint32_t)context.dagCache.numberItem };
    void* ptrDagNumberItemDag{ (void*)&dagNumberItemU32 };
    CUDA_ER(cudaMemcpyToSymbol(d_dag_number_item, ptrDagNumberItemDag, sizeof(uint32_t)));

    uint32_t const lightNnumberItem{ (uint32_t)context.lightCache.numberItem };
    void* ptrLightNumberItemDag{ (void*)&lightNnumberItem };
    CUDA_ER(cudaMemcpyToSymbol(d_light_number_item, ptrLightNumberItemDag, sizeof(uint32_t)));

    return true;
}


__host__
bool progpowUpdateConstants(
    uint32_t const* const hostBuffer,
    uint32_t* const deviceBuffer)
{
    CUDA_ER(cudaMemcpy(deviceBuffer,
                       hostBuffer,
                       algo::LEN_HASH_256_WORD_32 * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));

    return true;
}
