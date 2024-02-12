#include <cuda.h>
#include <cuda_runtime.h>

#include <common/error/cuda_error.hpp>
#include <device/nvidia.hpp>
#include <resolver/nvidia/nvidia.hpp>


device::DeviceNvidia::~DeviceNvidia()
{
    cleanUp();
}


bool device::DeviceNvidia::initialize()
{
    CU_ER(cuInit(0));
    CU_ER(cuDeviceGet(&cuDevice, cuIndex));
    CU_ER(cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice));
    CUDA_ER(cudaStreamCreateWithFlags(&cuStream, cudaStreamNonBlocking));

    resolver::ResolverNvidia* resolverNvidia{ dynamic_cast<resolver::ResolverNvidia*>(resolver) };
    if (nullptr != resolverNvidia)
    {
        resolverNvidia->cuStream = cuStream;
        resolverNvidia->cuProperties = &properties;
    }

    return true;
}


void device::DeviceNvidia::cleanUp()
{
    cudaError_t cuCodeError{ cudaDeviceReset() };
    if (cudaSuccess != cuCodeError)
    {
        logErr()
            << "cudaDeviceReset() failled: "
            << cudaGetErrorString(cuCodeError);
    }
}

