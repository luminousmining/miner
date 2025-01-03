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

        using NVMLInitFunc = nvmlReturn_t(*)();
        using NVMLShutdownFunc = nvmlReturn_t(*)();
        using NVMLDeviceGetHandleByIndexFunc = nvmlReturn_t(*)(unsigned int, nvmlDevice_t*);
        using NVMLDeviceGetPowerUsageFunc = nvmlReturn_t(*)(nvmlDevice_t, unsigned int*);
        using NVMLDeviceGetClockInfo = nvmlReturn_t(*)(nvmlDevice_t, nvmlClockType_t, unsigned int*);
        using NVMLDeviceGetUtilizationRates = nvmlReturn_t(*)(void*, void*);

        NVMLInitFunc                   nvmlInit{ nullptr };
        NVMLShutdownFunc               nvmlShutdown{ nullptr };
        NVMLDeviceGetHandleByIndexFunc nvmlDeviceGetHandleByIndex{ nullptr };
        NVMLDeviceGetPowerUsageFunc    nvmlDeviceGetPowerUsage{ nullptr };
        NVMLDeviceGetClockInfo         nvmlDeviceGetClockInfo{ nullptr };
        NVMLDeviceGetUtilizationRates  nvmlDeviceGetUtilizationRates{ nullptr };

        void* loadFunction(char const* name);
    };
}

#endif //!CUDA_ENABLE
