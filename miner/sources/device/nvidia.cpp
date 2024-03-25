#include <cuda.h>
#include <cuda_runtime.h>

#include <common/cast.hpp>
#include <common/config.hpp>
#include <common/custom.hpp>
#include <common/error/cuda_error.hpp>
#include <resolver/nvidia/nvidia.hpp>


bool device::DeviceNvidia::initialize()
{
    common::Config const& config { common::Config::instance() };
    resolver::ResolverNvidia* resolverNvidia{ dynamic_cast<resolver::ResolverNvidia*>(resolver) };
    std::optional<common::Config::DeviceOptionsDev> devConfig{ config.getConfigDev(id) };

    ////////////////////////////////////////////////////////////////////////////
    CUDA_ER(cudaSetDevice(cuIndex));

    ////////////////////////////////////////////////////////////////////////////
    cleanUp();

    ////////////////////////////////////////////////////////////////////////////
    if (nullptr == resolverNvidia)
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    CU_ER(cuInit(0));
    CU_ER(cuDeviceGet(&cuDevice, cuIndex));
    CU_ER(cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice));

    ////////////////////////////////////////////////////////////////////////////
    resolverNvidia->cuProperties = &properties;

    ////////////////////////////////////////////////////////////////////////////
    if (std::nullopt != devConfig)
    {
        resolverNvidia->isDoubleStream = (*devConfig).doubleStream;
    }
    if (false == resolverNvidia->isDoubleStream)
    {
        CUDA_ER(cudaStreamCreateWithFlags(&cuStream[0], cudaStreamNonBlocking));
        resolverNvidia->cuStream[0] = cuStream[0];
    }
    else
    {
        for (size_t i { 0u }; i < device::DeviceNvidia::MAX_INDEX_STREAM; ++i)
        {
            CUDA_ER(cudaStreamCreateWithFlags(&cuStream[i], cudaStreamNonBlocking));
            resolverNvidia->cuStream[i] = cuStream[i];
        }
    }

    return true;
}


void device::DeviceNvidia::cleanUp()
{
    CUresult cuResult { CUresult::CUDA_SUCCESS };
    cudaError_t cuCodeError { cudaSuccess };

    for (size_t i{ 0u }; i < device::DeviceNvidia::MAX_INDEX_STREAM; ++i)
    {
        if (nullptr != cuStream[i])
        {
            cuCodeError = cudaStreamDestroy(cuStream[i]);
            if (cudaSuccess != cuCodeError)
            {
                logErr()
                    << "cudaStreamDestroy(" << i << ") failled: "
                    << cudaGetErrorString(cuCodeError);
            }
            else
            {
                cuStream[i] = nullptr;
            }
        }
    }

    if (nullptr != cuContext)
    {
        cuResult = cuCtxDestroy(cuContext);
        if (CUresult::CUDA_SUCCESS != cuResult)
        {
            char const* msg;
            cuGetErrorString(cuResult, &msg);
            logErr()
                << "cuCtxDestroy() failled: "
                << msg;
        }
        else
        {
            cuContext = nullptr;
        }
    }

    cuCodeError = cudaDeviceReset();
    if (cudaSuccess != cuCodeError)
    {
        logErr()
            << "cudaDeviceReset() failled: "
            << cudaGetErrorString(cuCodeError);
    }
}
