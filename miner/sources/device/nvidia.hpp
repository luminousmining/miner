#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <device/device.hpp>


namespace device
{
    class DeviceNvidia : public device::Device
    {
    public:
        ~DeviceNvidia();

        uint32_t       cuIndex{ 0u };
        cudaDeviceProp properties;
        CUdevice       cuDevice;
        CUcontext      cuContext{ nullptr };
        cudaStream_t   cuStream{ nullptr };

    protected:
        bool initialize() final;
        void cleanUp() final;
    };
}
