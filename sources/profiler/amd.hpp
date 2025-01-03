#pragma once

#if defined(AMD_ENABLE)

#include <profiler/adl/adl_sdk.h>


namespace profiler
{
    struct Amd
    {
    public:
        bool valid{ false };

        bool load();
        void unload();
        ADLPMActivity getCurrentActivity(uint32_t const id);

    private:
#if defined(_WIN32)
        HMODULE libModule{ nullptr };
#else
        void* libModule{ nullptr };
#endif

        using ADL_MAIN_CONTROL_CREATE = int(*)(ADL_MAIN_MALLOC_CALLBACK callback, int enumConnectedAdapters);
        using ADL_MAIN_CONTROL_DESTROY = int(*)();
        using ADL_PM_ACTIVITY_GET = int(*)(int iAdapterIndex, ADLPMActivity* lpActivity);

        ADL_MAIN_CONTROL_CREATE  adlMainControlCreate{ nullptr };
        ADL_MAIN_CONTROL_DESTROY adlMainControlDestroy{ nullptr };
        ADL_PM_ACTIVITY_GET      adlOverdrive5CurrentActivityGet{ nullptr };

        void* loadFunction(char const* name);
    };
}

#endif //!CUDA_ENABLE
