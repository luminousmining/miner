#pragma once


__host__
bool kheavyhashFreeMemory(
    resolver::nvidia::kheavyhash::KernelParameters& params)
{
    CU_SAFE_DELETE(params.matrix);
    CU_SAFE_DELETE(params.header);
    CU_SAFE_DELETE(params.target);
    CU_SAFE_DELETE_HOST(params.resultCache);
    return true;
}


__host__
bool kheavyhashInitMemory(
    resolver::nvidia::kheavyhash::KernelParameters& params)
{
    ////////////////////////////////////////////////////////////////////////////
    if (false == kheavyhashFreeMemory(params))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    CU_ALLOC(&params.matrix, 64u * 64u * sizeof(uint16_t));
    CU_ALLOC(&params.header, algo::LEN_HASH_256);
    CU_ALLOC(&params.target, algo::LEN_HASH_256);
    CU_ALLOC_HOST(&params.resultCache, sizeof(algo::kheavyhash::Result) * 2u);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


__host__
bool kheavyhashUpdateConstants(
    resolver::nvidia::kheavyhash::KernelParameters& params)
{
    CUDA_ER(cudaMemcpy(params.matrix,
                       params.hostMatrix,
                       64u * 64u * sizeof(uint16_t),
                       cudaMemcpyHostToDevice));

    CUDA_ER(cudaMemcpy(params.header->ubytes,
                       params.hostHeader.ubytes,
                       algo::LEN_HASH_256,
                       cudaMemcpyHostToDevice));

    CUDA_ER(cudaMemcpy(params.target->ubytes,
                       params.hostTarget.ubytes,
                       algo::LEN_HASH_256,
                       cudaMemcpyHostToDevice));

    return true;
}
