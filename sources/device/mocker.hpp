#pragma once

#if defined(TOOL_MOCKER)


#include <device/device.hpp>


namespace device
{
    struct DeviceMocker : public device::Device
    {
    protected:
        bool initialize() final;
        void cleanUp() final;
    };
}

#endif
