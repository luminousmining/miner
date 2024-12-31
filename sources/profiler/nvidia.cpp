#if defined(CUDA_ENABLE)

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <cmath>

#include <common/cast.hpp>
#include <common/error/nvml_error.hpp>
#include <common/log/log.hpp>
#include <profiler/nvidia.hpp>


bool profiler::Nvidia::load()
{
#ifdef _WIN32
    libModule = LoadLibrary("nvml.dll");
#else
    libModule = dlopen("libnvidia-ml.so", RTLD_LAZY);
#endif

    if (nullptr == libModule)
    {
        logErr() << "Cannot load nvml library!";
        return false;
    }

    nvmlInit = reinterpret_cast<NVMLInitFunc>(loadFunction("nvmlInit"));
    nvmlShutdown = reinterpret_cast<NVMLShutdownFunc>(loadFunction("nvmlShutdown"));
    nvmlDeviceGetHandleByIndex = reinterpret_cast<NVMLDeviceGetHandleByIndexFunc>(loadFunction("nvmlDeviceGetHandleByIndex"));
    nvmlDeviceGetPowerUsage = reinterpret_cast<NVMLDeviceGetPowerUsageFunc>(loadFunction("nvmlDeviceGetPowerUsage"));

    return true;
}


void profiler::Nvidia::unload()
{
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


bool profiler::Nvidia::init(
    uint32_t const id,
    nvmlDevice_t* device)
{
    if (nullptr == nvmlInit)
    {
        logErr() << "nvmlInit is nullptr!";
        return false;
    }

    NVML_ER(nvmlInit());
    NVML_ER(nvmlDeviceGetHandleByIndex(id, device));

    return true;
}


double profiler::Nvidia::getPowerUsage(nvmlDevice_t device)
{
    uint32_t power{ 0u };
    if (nullptr == device)
    {
        logErr() << "nvml device is nullptr";
        return 0.0;
    }

    NVML_CALL(nvmlDeviceGetPowerUsage(device, &power));
    return  castDouble(power) / 1000.0;
}


void* profiler::Nvidia::loadFunction(char const* name)
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

#endif //!CUDA_ENABLE
