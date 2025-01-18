#pragma once

#if defined(CUDA_ENABLE)

#include <nvml.h>


namespace profiler
{
    struct Nvidia
    {
    public:
        bool valid{ false };

        bool load();
        void unload();
        bool init(uint32_t const id, nvmlDevice_t* device);
        double getPowerUsage(nvmlDevice_t device);
        uint32_t getCoreClock(nvmlDevice_t device);
        uint32_t getMemoryClock(nvmlDevice_t device);
        uint32_t getUtilizationRate(nvmlDevice_t device);

    private:
#if defined(_WIN32)
        HMODULE libModule{ nullptr };
#else
        void* libModule{ nullptr };
#endif

        using NVMLInit = nvmlReturn_t(*)();
        using NVMLShutdown = nvmlReturn_t(*)();
        using NVMLDeviceGetHandleByIndex = nvmlReturn_t(*)(unsigned int, nvmlDevice_t*);
        using NVMLDeviceGetPowerUsage = nvmlReturn_t(*)(nvmlDevice_t, unsigned int*);
        using NVMLDeviceGetClockInfo = nvmlReturn_t(*)(nvmlDevice_t, nvmlClockType_t, unsigned int*);
        using NVMLDeviceGetUtilizationRates = nvmlReturn_t(*)(void*, void*);
        using NVMLErrorString = const char*(*)(nvmlReturn_t);

        NVMLInit                       nvmlInit{ nullptr };
        NVMLShutdown                   nvmlShutdown{ nullptr };
        NVMLDeviceGetHandleByIndex     nvmlDeviceGetHandleByIndex{ nullptr };
        NVMLDeviceGetPowerUsage        nvmlDeviceGetPowerUsage{ nullptr };
        NVMLDeviceGetClockInfo         nvmlDeviceGetClockInfo{ nullptr };
        NVMLDeviceGetUtilizationRates  nvmlDeviceGetUtilizationRates{ nullptr };
        NVMLErrorString                nvmlErrorString{ nullptr };

        void* loadFunction(char const* name);
    };
}

#endif //!CUDA_ENABLE
