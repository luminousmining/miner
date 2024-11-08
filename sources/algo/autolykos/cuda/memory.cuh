#pragma once


__host__
bool autolykosv2FreeMemory(
    resolver::nvidia::autolykos_v2::KernelParameters& params)
{
    CU_SAFE_DELETE(params.header);
    CU_SAFE_DELETE(params.dag);
    CU_SAFE_DELETE(params.BHashes);
    CU_SAFE_DELETE_HOST(params.resultCache);
    return true;
}


__host__
bool autolykosv2InitMemory(
    resolver::nvidia::autolykos_v2::KernelParameters& params)
{
    ////////////////////////////////////////////////////////////////////////////
    if (false == autolykosv2FreeMemory(params))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    CU_ALLOC(&params.header, algo::LEN_HASH_256);
    CU_ALLOC(&params.dag, params.hostDagItemCount * algo::LEN_HASH_256);
    CU_ALLOC(&params.BHashes, algo::autolykos_v2::NONCES_PER_ITER * algo::LEN_HASH_256);
    CU_ALLOC_HOST(&params.resultCache, sizeof(algo::autolykos_v2::Result));

    ////////////////////////////////////////////////////////////////////////////
    IS_NULL(params.header);
    IS_NULL(params.dag);
    IS_NULL(params.BHashes);
    IS_NULL(params.resultCache);

    ////////////////////////////////////////////////////////////////////////////
    params.resultCache->count = 0u;
    params.resultCache->found = false;
    for (uint32_t i{ 0u }; i < 4u; ++i)
    {
        params.resultCache->nonces[i] = 0ull;
    }

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


__host__
bool autolykosv2UpateConstants(
    resolver::nvidia::autolykos_v2::KernelParameters& params)
{
    CUDA_ER(cudaMemcpy(params.header->ubytes,
                       params.hostHeader.ubytes,
                       algo::LEN_HASH_256,
                       cudaMemcpyHostToDevice));

    CUDA_ER(cudaMemcpyToSymbol(d_bound,
                               (void*)&params.hostBoundary,
                               algo::LEN_HASH_256));

    return true;
}
