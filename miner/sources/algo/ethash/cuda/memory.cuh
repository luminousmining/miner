#pragma once

#include <common/custom.hpp>


__host__
bool ethashFreeMemory(
    resolver::nvidia::ethash::KernelParameters& params)
{
    CU_SAFE_DELETE(params.lightCache);
    CU_SAFE_DELETE(params.dagCache);
    CU_SAFE_DELETE_HOST(params.resultCache);
    return true;
}


__host__
bool ethashInitMemory(
    algo::DagContext const& context,
    resolver::nvidia::ethash::KernelParameters& params)
{
    ////////////////////////////////////////////////////////////////////////////
    if (false == ethashFreeMemory(params))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    CUDA_ER(cudaMalloc((void**)&params.lightCache, context.lightCache.size));
    CUDA_ER(cudaMalloc((void**)&params.dagCache, context.dagCache.size));
    CUDA_ER(cudaMallocHost((void**)&params.resultCache, sizeof(algo::ethash::Result) * 2, 0));

    ////////////////////////////////////////////////////////////////////////////
    CUDA_ER(cudaMemcpy((void*)params.lightCache,
                       (void const*)context.lightCache.hash->bytes,
                       context.lightCache.size,
                       cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////
    IS_NULL(params.lightCache);
    IS_NULL(params.dagCache);
    IS_NULL(params.resultCache);

    ////////////////////////////////////////////////////////////////////////////
    params.resultCache[0].found = false;
    params.resultCache[0].count = 0u;
    params.resultCache[0].nonces[0] = 0ull;
    params.resultCache[0].nonces[1] = 0ull;
    params.resultCache[0].nonces[2] = 0ull;
    params.resultCache[0].nonces[3] = 0ull;
    params.resultCache[1].found = false;
    params.resultCache[1].count = 0u;
    params.resultCache[1].nonces[0] = 0ull;
    params.resultCache[1].nonces[1] = 0ull;
    params.resultCache[1].nonces[2] = 0ull;
    params.resultCache[1].nonces[3] = 0ull;

    ////////////////////////////////////////////////////////////////////////////
    CUDA_ER(cudaMemcpyToSymbol(d_light_cache, (void**)&params.lightCache, sizeof(uint4*)));
    CUDA_ER(cudaMemcpyToSymbol(d_dag, (void**)&params.dagCache, sizeof(uint4*)));

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const dagNumberItemU32{ (uint32_t)context.dagCache.numberItem };
    void* ptrDagNumberItemDag{ (void*)&dagNumberItemU32 };
    CUDA_ER(cudaMemcpyToSymbol(d_dag_number_item, ptrDagNumberItemDag, sizeof(uint32_t)));

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const lightNnumberItem{ (uint32_t)context.lightCache.numberItem };
    void* ptrLightNumberItemDag{ (void*)&lightNnumberItem };
    CUDA_ER(cudaMemcpyToSymbol(d_light_number_item, ptrLightNumberItemDag, sizeof(uint32_t)));

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


__host__
bool ethashUpdateConstants(
    uint32_t const* const header,
    uint64_t const boundary)
{
    uint4 const* const headerU4{ (uint4 const* const)header };
    CUDA_ER(cudaMemcpyToSymbol(d_header, headerU4, sizeof(uint4) * 2));
    CUDA_ER(cudaMemcpyToSymbol(d_boundary, (void*)&boundary, sizeof(uint64_t)));

    return true;
}
