#pragma once

#include <common/error/cuda_error.hpp>


namespace benchmark
{
    struct PropertiesNvidia
    {
        uint32_t       cuIndex { 0u };
        CUdevice       cuDevice;
        CUcontext      cuContext{ nullptr };
        cudaStream_t   cuStream{ nullptr };
        cudaDeviceProp cuProperties{};
    };

    bool cleanUpCuda();
    bool initializeCuda(benchmark::PropertiesNvidia& properties,
                        uint32_t const index = 0u);
}
