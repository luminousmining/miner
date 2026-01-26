#if defined(CUDA_ENABLE)

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <cmath>
#include <set>

#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/error/nvml_error.hpp>
#include <common/log/log.hpp>
#include <profiler/nvidia.hpp>


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


bool profiler::Nvidia::load()
{
#ifdef _WIN32
    libModule = LoadLibrary("nvml.dll");
#else
    std::set<std::string> allPathLibNvidiaML
    {
        "libnvidia-ml.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        "/usr/lib64/libnvidia-ml.so.1",
        "/usr/lib/libnvidia-ml.so.1"
    };
    for (auto pathLibNvidiaML : allPathLibNvidiaML)
    {
        libModule = dlopen(pathLibNvidiaML.c_str(), RTLD_LAZY);
        if (nullptr != libModule)
        {
            logInfo() << "Loaded library " << pathLibNvidiaML;
            break;
        }
    }
#endif

    if (nullptr == libModule)
    {
        logErr() << "Cannot load nvml library!";
        return false;
    }

    nvmlInit = reinterpret_cast<NVMLInit>(loadFunction("nvmlInit"));
    nvmlShutdown = reinterpret_cast<NVMLShutdown>(loadFunction("nvmlShutdown"));
    nvmlDeviceGetHandleByIndex = reinterpret_cast<NVMLDeviceGetHandleByIndex>(loadFunction("nvmlDeviceGetHandleByIndex"));
    nvmlDeviceGetPowerUsage = reinterpret_cast<NVMLDeviceGetPowerUsage>(loadFunction("nvmlDeviceGetPowerUsage"));
    nvmlDeviceGetClockInfo = reinterpret_cast<NVMLDeviceGetClockInfo>(loadFunction("nvmlDeviceGetClockInfo"));
    nvmlDeviceGetUtilizationRates = reinterpret_cast<NVMLDeviceGetUtilizationRates>(loadFunction("nvmlDeviceGetUtilizationRates"));
    nvmlErrorString = reinterpret_cast<NVMLErrorString>(loadFunction("nvmlErrorString"));

    IS_NULL(nvmlInit);
    IS_NULL(nvmlShutdown);
    IS_NULL(nvmlDeviceGetHandleByIndex);
    IS_NULL(nvmlDeviceGetPowerUsage);
    IS_NULL(nvmlDeviceGetClockInfo);
    IS_NULL(nvmlDeviceGetUtilizationRates);
    IS_NULL(nvmlErrorString);

    valid = true;

    return true;
}


void profiler::Nvidia::unload()
{
    if (nullptr != nvmlShutdown)
    {
        nvmlShutdown();
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


bool profiler::Nvidia::init(
    uint32_t const id,
    nvmlDevice_t* device)
{
    NVML_ER(nvmlInit());
    NVML_ER(nvmlDeviceGetHandleByIndex((unsigned int)id, device));
    return true;
}


double profiler::Nvidia::getPowerUsage(nvmlDevice_t device)
{
    uint32_t power{ 0u };
    NVML_CALL(nvmlDeviceGetPowerUsage(device, &power));
    return castDouble(power) / 1000.0;
}


uint32_t profiler::Nvidia::getCoreClock(nvmlDevice_t device)
{
    uint32_t clock{ 0u };
    NVML_CALL(nvmlDeviceGetClockInfo(device, nvmlClockType_t::NVML_CLOCK_MEM, &clock));
    return clock;
}


uint32_t profiler::Nvidia::getMemoryClock(nvmlDevice_t device)
{
    uint32_t clock{ 0u };
    NVML_CALL(nvmlDeviceGetClockInfo(device, nvmlClockType_t::NVML_CLOCK_GRAPHICS, &clock));
    return clock;
}


uint32_t profiler::Nvidia::getUtilizationRate(nvmlDevice_t device)
{
    nvmlUtilization_t utilization{};
    NVML_CALL(nvmlDeviceGetUtilizationRates(device, &utilization));
    return utilization.gpu;
}

#endif
