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
#if defined(TOOL_MOCKER)
        MOCKER,
#endif
        UNKNOWN
    };

    constexpr uint8_t MAX_DEVICE_TYPE{ 2 };
}
