#if defined(AMD_ENABLE)

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <vector>

#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/log/log.hpp>
#include <profiler/amd.hpp>


////////////////////////////////////////////////////////////////////////////////
/// AMD's PCI vendor ID. Used to skip non-AMD adapters when ADL enumerates the
/// whole system (e.g. an Intel/NVIDIA card sharing the rig).
static constexpr int AMD_VENDOR_ID{ 1002 };


void* profiler::Amd::loadFunction(char const* name, bool const required)
{
#ifdef _WIN32
    void* ptr{ castVOIDP(GetProcAddress(libModule, name)) };
    if (nullptr == ptr && true == required) [[unlikely]]
    {
        logErr() << "Cannot load function: " << name << " (" << GetLastError() << ")";
    }
#else
    void* ptr{ castVOIDP(dlsym(libModule, name)) };
    if (nullptr == ptr && true == required) [[unlikely]]
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
    if (nullptr == libModule) [[unlikely]]
    {
        libModule = LoadLibrary("atiadlxy.dll"); // 32-bits
    }
#else
    libModule = dlopen("libatiadlxx.so", RTLD_LAZY);
#endif

    if (nullptr == libModule) [[unlikely]]
    {
        logErr() << "Cannot load ADL library";
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // ADL2 (modern) entry points. Optional: absent on very old drivers, in which
    // case we degrade to the legacy Overdrive5 path below.
    adl2MainControlCreate =
        reinterpret_cast<ADL2_MAIN_CONTROL_CREATE>(loadFunction("ADL2_Main_Control_Create", false));
    adl2MainControlDestroy =
        reinterpret_cast<ADL2_MAIN_CONTROL_DESTROY>(loadFunction("ADL2_Main_Control_Destroy", false));
    adl2AdapterNumberOfAdaptersGet =
        reinterpret_cast<ADL2_ADAPTER_NUMBEROFADAPTERS_GET>(loadFunction("ADL2_Adapter_NumberOfAdapters_Get", false));
    adl2AdapterAdapterInfoGet =
        reinterpret_cast<ADL2_ADAPTER_ADAPTERINFO_GET>(loadFunction("ADL2_Adapter_AdapterInfo_Get", false));
    adl2NewQueryPMLogDataGet =
        reinterpret_cast<ADL2_PMLOG_QUERY_GET>(loadFunction("ADL2_New_QueryPMLogData_Get", false));

    ////////////////////////////////////////////////////////////////////////////
    // ADL1 (legacy) entry points, used only for the Overdrive5 fallback.
    adlMainControlCreate =
        reinterpret_cast<ADL_MAIN_CONTROL_CREATE>(loadFunction("ADL_Main_Control_Create", false));
    adlMainControlDestroy =
        reinterpret_cast<ADL_MAIN_CONTROL_DESTROY>(loadFunction("ADL_Main_Control_Destroy", false));
    adlAdapterNumberOfAdaptersGet =
        reinterpret_cast<ADL_ADAPTER_NUMBEROFADAPTERS_GET>(loadFunction("ADL_Adapter_NumberOfAdapters_Get", false));
    adlAdapterAdapterInfoGet =
        reinterpret_cast<ADL_ADAPTER_ADAPTERINFO_GET>(loadFunction("ADL_Adapter_AdapterInfo_Get", false));
    adlOverdrive5CurrentActivityGet =
        reinterpret_cast<ADL_PM_ACTIVITY_GET>(loadFunction("ADL_Overdrive5_CurrentActivity_Get", false));

    ////////////////////////////////////////////////////////////////////////////
    // ADL's malloc callback. ADL uses it to allocate the buffers it hands back
    // (adapter info, etc), so it MUST honour the requested size — the previous
    // implementation returned a fixed 1-byte block, corrupting the heap the
    // moment ADL wrote a structure into it. A captureless lambda converts to the
    // ADL_MAIN_MALLOC_CALLBACK function pointer.
    ADL_MAIN_MALLOC_CALLBACK const adlMalloc{ [](int const size) -> void*
                                              { return malloc(static_cast<size_t>(size)); } };

    ////////////////////////////////////////////////////////////////////////////
    // Preferred path: ADL2 context + PMLog sensor query. Works on RDNA/Vega and
    // newer, which is everything the legacy Overdrive5 API cannot read.
    if (nullptr != adl2MainControlCreate && nullptr != adl2NewQueryPMLogDataGet)
    {
        if (ADL_OK == adl2MainControlCreate(adlMalloc, 1, &context) && nullptr != context)
        {
            if (true == buildAdapterMap())
            {
                backend = Backend::PMLOG;
                valid = true;
                logInfo() << "AMD telemetry backend: ADL2 PMLog (" << busToAdapter.size() << " adapter(s))";
                return true;
            }
            logErr() << "ADL2 found no AMD adapters; falling back to Overdrive5";
        }
        else
        {
            logErr() << "ADL2_Main_Control_Create failed; falling back to Overdrive5";
        }

        // ADL2 context unusable — tear it down before trying the legacy path.
        if (nullptr != context && nullptr != adl2MainControlDestroy)
        {
            adl2MainControlDestroy(context);
        }
        context = nullptr;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Legacy fallback: ADL1 context + Overdrive5 activity. Pre-GCN1.2 only.
    if (nullptr != adlMainControlCreate && nullptr != adlOverdrive5CurrentActivityGet)
    {
        if (ADL_OK == adlMainControlCreate(adlMalloc, 1))
        {
            buildAdapterMap(); // best-effort; Overdrive5 can still use raw indices
            backend = Backend::OVERDRIVE5;
            valid = true;
            logInfo() << "AMD telemetry backend: Overdrive5 (legacy)";
            return true;
        }
        logErr() << "ADL_Main_Control_Create failed";
    }

    logErr() << "No usable AMD telemetry backend (ADL2 PMLog / Overdrive5 unavailable)";
    return false;
}


void profiler::Amd::unload()
{
    if (Backend::OVERDRIVE5 == backend && nullptr != adlMainControlDestroy)
    {
        adlMainControlDestroy();
    }
    if (nullptr != context && nullptr != adl2MainControlDestroy)
    {
        adl2MainControlDestroy(context);
        context = nullptr;
    }
    if (nullptr != libModule)
    {
#ifdef _WIN32
        FreeLibrary(libModule);
#else
        dlclose(libModule);
#endif
        libModule = nullptr;
    }
}


////////////////////////////////////////////////////////////////////////////////
/// Enumerate ADL adapters and map each AMD GPU's PCI bus number to its ADL
/// adapter index. The ADL adapter index is neither the OpenCL device index nor
/// the PCI bus, so the only reliable cross-reference is the bus number, which the
/// device layer already tracks.
bool profiler::Amd::buildAdapterMap()
{
    busToAdapter.clear();

    int numAdapters{ 0 };

    bool const useAdl2{ nullptr != context && nullptr != adl2AdapterNumberOfAdaptersGet
                        && nullptr != adl2AdapterAdapterInfoGet };

    if (true == useAdl2)
    {
        if (ADL_OK != adl2AdapterNumberOfAdaptersGet(context, &numAdapters))
        {
            return false;
        }
    }
    else if (nullptr != adlAdapterNumberOfAdaptersGet && nullptr != adlAdapterAdapterInfoGet)
    {
        if (ADL_OK != adlAdapterNumberOfAdaptersGet(&numAdapters))
        {
            return false;
        }
    }
    else
    {
        return false;
    }

    if (0 >= numAdapters)
    {
        return false;
    }

    // Value-initialised so ADL sees zeroed structures.
    std::vector<AdapterInfo> info(static_cast<size_t>(numAdapters), AdapterInfo{});
    int const                inputSize{ static_cast<int>(sizeof(AdapterInfo) * static_cast<size_t>(numAdapters)) };

    int const result{ true == useAdl2 ? adl2AdapterAdapterInfoGet(context, info.data(), inputSize)
                                       : adlAdapterAdapterInfoGet(info.data(), inputSize) };
    if (ADL_OK != result)
    {
        return false;
    }

    for (AdapterInfo const& adapter : info)
    {
        if (AMD_VENDOR_ID != adapter.iVendorID || 0 == adapter.iPresent)
        {
            continue;
        }
        // One physical GPU can expose several ADL indices; keep the first per bus.
        busToAdapter.emplace(static_cast<uint32_t>(adapter.iBusNumber), adapter.iAdapterIndex);
    }

    return false == busToAdapter.empty();
}


////////////////////////////////////////////////////////////////////////////////
/// Translate a device PCI bus number into an ADL adapter index. If the map only
/// holds a single AMD GPU, accept it regardless of bus so a mismatch in bus
/// numbering still yields telemetry on single-card rigs.
bool profiler::Amd::resolveAdapterIndex(uint32_t const pciBus, int& adapterIndex)
{
    auto const it{ busToAdapter.find(pciBus) };
    if (it != busToAdapter.end())
    {
        adapterIndex = it->second;
        return true;
    }

    if (1 == busToAdapter.size())
    {
        adapterIndex = busToAdapter.begin()->second;
        return true;
    }

    return false;
}


profiler::Amd::Telemetry profiler::Amd::getTelemetry(uint32_t const pciBus)
{
    Telemetry telemetry{};

    if (false == valid)
    {
        return telemetry;
    }

    int adapterIndex{ 0 };
    if (false == resolveAdapterIndex(pciBus, adapterIndex))
    {
        // Overdrive5 has no adapter map on most paths; fall back to the bus as
        // index to preserve historical single-GPU behaviour.
        if (Backend::OVERDRIVE5 != backend)
        {
            if (true == warnedBuses.insert(pciBus).second)
            {
                logWarn() << "No ADL adapter matches PCI bus " << pciBus
                          << "; telemetry will read zero for this device";
            }
            return telemetry;
        }
        adapterIndex = static_cast<int>(pciBus);
    }

    switch (backend)
    {
        case Backend::PMLOG:
        {
            return getTelemetryPMLog(adapterIndex);
        }
        case Backend::OVERDRIVE5:
        {
            return getTelemetryOverdrive5(adapterIndex);
        }
        case Backend::NONE:
        {
            break;
        }
    }

    return telemetry;
}


////////////////////////////////////////////////////////////////////////////////
/// Read live sensors via ADL2 PMLog. Each sensor is indexed by its
/// ADL_PMLOG_SENSORS enum value and carries a `supported` flag.
profiler::Amd::Telemetry profiler::Amd::getTelemetryPMLog(int const adapterIndex)
{
    Telemetry telemetry{};

    ADLPMLogDataOutput data{};
    if (ADL_OK != adl2NewQueryPMLogDataGet(context, adapterIndex, &data))
    {
        logErr() << "ADL PMLog query failed (adapter " << adapterIndex << ")";
        return telemetry;
    }

    auto const sensor{ [&data](ADL_PMLOG_SENSORS const id) -> int
                       { return 0 != data.sensors[id].supported ? data.sensors[id].value : 0; } };

    telemetry.coreClock = static_cast<uint32_t>(sensor(ADL_PMLOG_CLK_GFXCLK));
    telemetry.memoryClock = static_cast<uint32_t>(sensor(ADL_PMLOG_CLK_MEMCLK));
    telemetry.utilization = static_cast<uint32_t>(sensor(ADL_PMLOG_INFO_ACTIVITY_GFX));

    // Power (W): prefer whole-board, then ASIC, then GFX-domain.
    int power{ sensor(ADL_PMLOG_BOARD_POWER) };
    if (0 == power)
    {
        power = sensor(ADL_PMLOG_ASIC_POWER);
    }
    if (0 == power)
    {
        power = sensor(ADL_PMLOG_GFX_POWER);
    }
    telemetry.power = static_cast<double>(power);

    return telemetry;
}


////////////////////////////////////////////////////////////////////////////////
/// Legacy Overdrive5 activity read. Clocks are reported in units of 10 kHz.
profiler::Amd::Telemetry profiler::Amd::getTelemetryOverdrive5(int const adapterIndex)
{
    Telemetry telemetry{};

    ADLPMActivity activity{};
    activity.iSize = sizeof(ADLPMActivity);

    if (ADL_OK != adlOverdrive5CurrentActivityGet(adapterIndex, &activity))
    {
        logErr() << "ADL cannot get activity (adapter " << adapterIndex << ")";
        return telemetry;
    }

    telemetry.coreClock = static_cast<uint32_t>(activity.iEngineClock) / 100u; // 10 kHz -> MHz
    telemetry.memoryClock = static_cast<uint32_t>(activity.iMemoryClock) / 100u;
    telemetry.utilization = static_cast<uint32_t>(activity.iActivityPercent);

    return telemetry;
}

#endif
