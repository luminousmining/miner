#pragma once

#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <device/device.hpp>

namespace device
{
    class DeviceNvidia : public device::Device
    {
    public:
        uint32_t       cuIndex{ 0u };
        cudaDeviceProp properties;
        CUdevice       cuDevice;
        CUcontext      cuContext{ nullptr };
        cudaStream_t   cuStream[2u]{ nullptr, nullptr };

    protected:
        bool initialize() final;
        void cleanUp() final;
    };
}
#endif
