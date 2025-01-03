#pragma once

#if defined(AMD_ENABLE)

#include <profiler/adl/adl_sdk.h>


namespace profiler
{
    struct Amd
    {
    public:
        bool load();
        void unload();
        bool init();
        double getPowerUsage(uint32_t const id);

    private:
#if defined(_WIN32)
        HMODULE libModule{ nullptr };
#else
        void* libModule{ nullptr };
#endif

        using ADL_MAIN_CONTROL_CREATE = int(*)(ADL_MAIN_MALLOC_CALLBACK callback, int enumConnectedAdapters);
        using ADL_MAIN_CONTROL_DESTROY = int(*)();
        using ADL_PM_ACTIVITY_GET = int(*)(int iAdapterIndex, ADLPMActivity* lpActivity);
        using ADL2_OVERDRIVEN_PERFORMANCESTATUS_GET = int(*)(void* context, int iAdapterIndex, ADLODNPerformanceStatus* status);

        ADL_MAIN_CONTROL_CREATE  adlMainControlCreate{ nullptr };
        ADL_MAIN_CONTROL_DESTROY adlMainControlDestroy{ nullptr };
        ADL_PM_ACTIVITY_GET      adlOverdrive5CurrentActivityGet{ nullptr };
        ADL2_OVERDRIVEN_PERFORMANCESTATUS_GET adl2OverdrivenPerformanceStatusGet{ nullptr };

        void* loadFunction(char const* name);
    };
}

#endif //!CUDA_ENABLE
