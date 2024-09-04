#pragma once


namespace device
{
    enum class DEVICE_TYPE : uint8_t
    {
#if defined(CUDA_ENABLE)
        NVIDIA,
#endif
#if defined(AMD_ENABLE)
        AMD,
#endif
        UNKNOW
    };
}
