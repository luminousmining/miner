#pragma once

#if defined(CPU_ENABLE)


#include <device/device.hpp>


namespace device
{
    struct DeviceCpu : public device::Device
    {
      protected:
        bool     initialize() final;
        void     cleanUp() final;
        uint32_t getMinimumKernelExecuted() const final;
    };
}

#endif
