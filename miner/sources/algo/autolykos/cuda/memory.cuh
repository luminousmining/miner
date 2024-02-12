#pragma once


bool autolykosv2InitMemory(
    resolver::nvidia::autolykos_v2::KernelParameters& params)
{
    ////////////////////////////////////////////////////////////////////////////
    CU_SAFE_DELETE(params.header);
    CU_SAFE_DELETE(params.dag);
    CU_SAFE_DELETE(params.BHashes);
    CU_SAFE_DELETE_HOST(params.resultCache);

    ////////////////////////////////////////////////////////////////////////////
    CUDA_ER(cudaMalloc((void**)&params.header, algo::LEN_HASH_256));
    CUDA_ER(cudaMalloc((void**)&params.dag, params.hostDagItemCount * algo::LEN_HASH_256));
    CUDA_ER(cudaMalloc((void**)&params.BHashes, algo::autolykos_v2::NONCES_PER_ITER * algo::LEN_HASH_256));
    CUDA_ER(cudaMallocHost((void**)&params.resultCache, sizeof(algo::autolykos_v2::Result), 0));

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
