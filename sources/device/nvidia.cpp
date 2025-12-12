#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <common/custom.hpp>
#include <common/error/cuda_error.hpp>
#include <device/nvidia.hpp>
#include <resolver/nvidia/nvidia.hpp>


bool device::DeviceNvidia::initialize()
{
    cleanUp();

    CU_ER(cuInit(0));
    CU_ER(cuDeviceGet(&cuDevice, cuIndex));
    CU_ER(cuCtxCreate(&cuContext, nullptr, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice));
    CUDA_ER(cudaStreamCreateWithFlags(&cuStream[0], cudaStreamNonBlocking));
    CUDA_ER(cudaStreamCreateWithFlags(&cuStream[1], cudaStreamNonBlocking));

    resolver::ResolverNvidia* resolverNvidia{ dynamic_cast<resolver::ResolverNvidia*>(resolver) };
    if (nullptr != resolverNvidia)
    {
        resolverNvidia->cuStream[0] = cuStream[0];
        resolverNvidia->cuStream[1] = cuStream[1];
        resolverNvidia->cuProperties = &properties;
        resolverNvidia->cuDevice = &cuDevice;
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
#endif
