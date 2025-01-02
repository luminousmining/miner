#if defined(AMD_ENABLE)

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <cmath>

#include <common/cast.hpp>
#include <common/log/log.hpp>
#include <profiler/amd.hpp>


bool profiler::Amd::load()
{
    return true;
}


void profiler::Amd::unload()
{
    if (nullptr != adlMainControlDestroy)
    {
        adlMainControlDestroy();
    }
#ifdef _WIN32
    if (nullptr != libModule)
    {
        FreeLibrary(libModule);
    }
#else
    if (nullptr != libModule)
    {
        dlclose(libModule);
    }
#endif
}


bool profiler::Amd::init()
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

    auto cbAdlControlCreate{ [](int) -> void* { return malloc(1); } };
    if (ADL_OK != adlMainControlCreate(cbAdlControlCreate, 1))
    {
        return false;
    }

    return true;
}


double profiler::Amd::getPowerUsage(uint32_t const id)
{
    ADLPMActivity activity{};
    if (ADL_OK != adlOverdrive5CurrentActivityGet(id, &activity))
    {
        logErr() << "ADL cannot get activity";
    }
    return activity.iPowerConsumption / 1000.0;
}


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

#endif
