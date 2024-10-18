#pragma once


__host__
bool blake3FreeMemory(
    resolver::nvidia::blake3::KernelParameters& params)
{
    CU_SAFE_DELETE(params.header);
    CU_SAFE_DELETE_HOST(params.resultCache);
    return true;
}


__host__
bool blake3InitMemory(
    resolver::nvidia::blake3::KernelParameters& params)
{
    ////////////////////////////////////////////////////////////////////////////
    if (false == blake3FreeMemory(params))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    CUDA_ER(cudaMalloc((void**)&params.header, algo::LEN_HASH_3072));
    CUDA_ER(cudaMalloc((void**)&params.target, algo::LEN_HASH_256));
    CUDA_ER(cudaMallocHost((void**)&params.resultCache, sizeof(algo::blake3::Result), 0));

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


__host__
bool blake3UpateConstants(
    resolver::nvidia::blake3::KernelParameters& params)
{
    CUDA_ER(cudaMemcpy(params.header->ubytes,
                       params.hostHeaderBlob.ubytes,
                       algo::LEN_HASH_3072,
                       cudaMemcpyHostToDevice));

    CUDA_ER(cudaMemcpy(params.target->ubytes,
                       params.hostTargetBlob.ubytes,
                       algo::LEN_HASH_256,
                       cudaMemcpyHostToDevice));

    return true;
}
