#if defined(AMD_ENABLE)

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <cmath>

#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <profiler/amd.hpp>


void* profiler::Amd::loadFunction(char const* name)
{
#ifdef _WIN32
    void* ptr{ castVOIDP(GetProcAddress(libModule, name)) };
    if (nullptr == ptr)
    {
        logErr() << "Cannot load function: " << name << " (" << GetLastError() << ")";
    }
#else
    void* ptr{ castVOIDP(dlsym(libModule, name)) };
    if (nullptr == ptr)
    {
        logErr() << "Cannot load function: " << name << " (" << dlerror() << ")";
    }
#endif

    return ptr;
}


bool profiler::Amd::load()
{
#ifdef _WIN32
    libModule = LoadLibrary("atiadlxx.dll"); // 64-bits
    if (nullptr == libModule)
    {
        libModule = LoadLibrary("atiadlxy.dll"); // 32-bits
    }
#else
    libModule = dlopen("libatiadlxx.so", RTLD_LAZY);
#endif

    if (nullptr == libModule)
    {
        logErr() << "Cannot load ADL library";
        return false;
    }

    adlMainControlCreate = reinterpret_cast<ADL_MAIN_CONTROL_CREATE>(loadFunction("ADL_Main_Control_Create"));
    adlMainControlDestroy = reinterpret_cast<ADL_MAIN_CONTROL_DESTROY>(loadFunction("ADL_Main_Control_Destroy"));
    adlOverdrive5CurrentActivityGet = reinterpret_cast<ADL_PM_ACTIVITY_GET>(loadFunction("ADL_Overdrive5_CurrentActivity_Get"));

    IS_NULL(adlMainControlCreate);
    IS_NULL(adlMainControlDestroy);
    IS_NULL(adlOverdrive5CurrentActivityGet);

    auto cbAdlControlCreate{ [](int) -> void* { return malloc(1); } };
    if (ADL_OK != adlMainControlCreate(cbAdlControlCreate, 1))
    {
        return false;
    }

    valid = true;

    return true;
}


void profiler::Amd::unload()
{
    if (nullptr != adlMainControlDestroy)
    {
        adlMainControlDestroy();
    }
    if (nullptr != libModule)
    {
#ifdef _WIN32
        FreeLibrary(libModule);
#else
        dlclose(libModule);
#endif
    }
}


ADLPMActivity profiler::Amd::getCurrentActivity(uint32_t const id)
{
    ADLPMActivity activity{};
    activity.iSize = sizeof(ADLPMActivity);

    if (ADL_OK != adlOverdrive5CurrentActivityGet(id, &activity))
    {
        logErr() << "ADL cannot get activity";
    }

    return activity;
}

#endif
