#if defined(CUDA_ENABLE)

#include <benchmark/nvidia.hpp>
#include <common/error/cuda_error.hpp>
#include <common/log/log.hpp>


bool benchmark::cleanUpCuda()
{
    CUDA_ER(cudaDeviceReset());
    return true;
}


bool benchmark::initializeCuda(
        benchmark::PropertiesNvidia& properties,
        uint32_t const index)
{
    if (false == cleanUpCuda())
    {
        return false;
    }

    properties.cuIndex = index;

    CU_ER(cuInit(0));
    CU_ER(cuDeviceGet(&properties.cuDevice, properties.cuIndex));
    CU_ER(cuCtxCreate(&properties.cuContext, CU_CTX_SCHED_BLOCKING_SYNC, properties.cuDevice));
    CUDA_ER(cudaStreamCreateWithFlags(&properties.cuStream, cudaStreamNonBlocking));
    CUDA_ER(cudaGetDeviceProperties(&properties.cuProperties, properties.cuIndex));

    logInfo() << "Device [" << properties.cuProperties.name << "] selected!";

    return true;
}

#endif
