#pragma once

#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <device/device.hpp>


namespace device
{
    class DeviceAmd : public device::Device
    {
    public:
        cl::Device clDevice;

    protected:
        cl::Context      clContext{};
        cl::CommandQueue clQueue{};

        bool initialize() final;
        void cleanUp() final;
    };
}
#endif
