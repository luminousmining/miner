#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <device/device.hpp>


namespace device
{
    class DeviceNvidia : public device::Device
    {
    public:
        static constexpr size_t MAX_INDEX_STREAM { 2 };

        uint32_t       cuIndex{ 0u };
        cudaDeviceProp properties;
        CUdevice       cuDevice { 0 };
        CUcontext      cuContext{ nullptr};
        cudaStream_t   cuStream[MAX_INDEX_STREAM]{ nullptr, nullptr };

    protected:
        bool initialize() final;
        void cleanUp() final;
    };
}
