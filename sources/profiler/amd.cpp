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


////////////////////////////////////////////////////////////////////////////////
/// Resolve the ADL2 (modern, context-based) entry points. Returns true only when
/// the full set required to drive the PMLog path is present, so the caller can
/// commit to Backend::PMLOG without re-checking individual pointers afterwards.
bool profiler::Amd::loadAdl2Symbols()
{
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

    return nullptr != adl2MainControlCreate && nullptr != adl2NewQueryPMLogDataGet
           && nullptr != adl2AdapterNumberOfAdaptersGet && nullptr != adl2AdapterAdapterInfoGet;
}


////////////////////////////////////////////////////////////////////////////////
/// Resolve the ADL1 (legacy, context-less) entry points for the Overdrive5
/// fallback. Only called when ADL2 is unavailable, so these symbols are never
/// looked up on modern drivers.
bool profiler::Amd::loadAdl1Symbols()
{
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

    return nullptr != adlMainControlCreate && nullptr != adlOverdrive5CurrentActivityGet;
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
    // ADL's malloc callback. ADL uses it to allocate the buffers it hands back
    // (adapter info, etc), so it MUST honour the requested size — the previous
    // implementation returned a fixed 1-byte block, corrupting the heap the
    // moment ADL wrote a structure into it. A captureless lambda converts to the
    // ADL_MAIN_MALLOC_CALLBACK function pointer.
    ADL_MAIN_MALLOC_CALLBACK const adlMalloc{ [](int const size) -> void* { return malloc(castSize(size)); } };

    ////////////////////////////////////////////////////////////////////////////
    // Preferred path: ADL2 context + PMLog sensor query. Works on RDNA/Vega and
    // newer, which is everything the legacy Overdrive5 API cannot read. The ADL1
    // symbols below are only resolved if this path is unavailable.
    if (true == loadAdl2Symbols())
    {
        int const createStatus{ adl2MainControlCreate(adlMalloc, 1, &context) };
        if (ADL_OK == createStatus && nullptr != context)
        {
            backend = Backend::PMLOG;
            if (true == buildAdapterMap())
            {
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
        backend = Backend::NONE;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Legacy fallback: ADL1 context + Overdrive5 activity. Pre-GCN1.2 only.
    if (true == loadAdl1Symbols())
    {
        if (ADL_OK == adlMainControlCreate(adlMalloc, 1))
        {
            backend = Backend::OVERDRIVE5;
            buildAdapterMap(); // best-effort; Overdrive5 can still use raw indices
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
    if (Backend::PMLOG == backend && nullptr != context && nullptr != adl2MainControlDestroy)
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

    // Dispatch on the backend the caller already committed to, not on raw
    // pointers: PMLOG always means the ADL2 context-based calls, OVERDRIVE5 the
    // ADL1 context-less ones.
    switch (backend)
    {
        case Backend::PMLOG:
        {
            if (ADL_OK != adl2AdapterNumberOfAdaptersGet(context, &numAdapters))
            {
                return false;
            }
            break;
        }
        case Backend::OVERDRIVE5:
        {
            // ADL1 adapter enumeration is optional (not part of loadAdl1Symbols'
            // required set): if absent, fail the map so getTelemetry falls back to
            // the legacy bus-as-index behaviour rather than calling a null pointer.
            if (nullptr == adlAdapterNumberOfAdaptersGet || nullptr == adlAdapterAdapterInfoGet
                || ADL_OK != adlAdapterNumberOfAdaptersGet(&numAdapters))
            {
                return false;
            }
            break;
        }
        case Backend::NONE:
        {
            return false;
        }
    }

    if (0 >= numAdapters)
    {
        return false;
    }

    std::vector<AdapterInfo> info(castSize(numAdapters), AdapterInfo{});
    int const                inputSize{ cast32(sizeof(AdapterInfo) * castSize(numAdapters)) };

    int const result{ Backend::PMLOG == backend ? adl2AdapterAdapterInfoGet(context, info.data(), inputSize)
                                                 : adlAdapterAdapterInfoGet(info.data(), inputSize) };
    if (ADL_OK != result)
    {
        return false;
    }

    for (AdapterInfo const& adapter : info)
    {
        // ADL returns iVendorID signed, and integrated AMD adapters (APUs) report
        // it negated (-1002 instead of 1002). Match either sign, otherwise the
        // iGPU is left out of the map and borrows the dGPU's telemetry through the
        // single-adapter fallback in resolveAdapterIndex().
        if ((AMD_VENDOR_ID != adapter.iVendorID && -AMD_VENDOR_ID != adapter.iVendorID)
            || 0 == adapter.iPresent)
        {
            continue;
        }
        // One physical GPU can expose several ADL indices; keep the first per bus.
        busToAdapter.emplace(castU32(adapter.iBusNumber), adapter.iAdapterIndex);
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

    if (1u == busToAdapter.size())
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
        switch (backend)
        {
            case Backend::OVERDRIVE5:
            {
                // Overdrive5 has no adapter map on most paths; fall back to the
                // bus as index to preserve historical single-GPU behaviour.
                adapterIndex = cast32(pciBus);
                break;
            }
            case Backend::PMLOG:
            case Backend::NONE:
            {
                bool const firstWarning{ warnedBuses.insert(pciBus).second };
                if (true == firstWarning)
                {
                    logWarn() << "No ADL adapter matches PCI bus " << pciBus
                              << "; telemetry will read zero for this device";
                }
                return telemetry;
            }
        }
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
        // Some adapters (notably integrated GPUs / APUs) return ADL_ERR_NOT_SUPPORTED
        // here: they expose no PMLog sensors, so telemetry stays zero. Warn once per
        // adapter instead of every stats interval.
        bool const firstWarning{ warnedAdapters.insert(adapterIndex).second };
        if (true == firstWarning)
        {
            logWarn() << "ADL PMLog unavailable for adapter " << adapterIndex
                      << "; telemetry will read zero for this device";
        }
        return telemetry;
    }

    auto const sensor{ [&data](ADL_PMLOG_SENSORS const id) -> int
                       { return 0 != data.sensors[id].supported ? data.sensors[id].value : 0; } };

    telemetry.coreClock = castU32(sensor(ADL_PMLOG_CLK_GFXCLK));
    telemetry.memoryClock = castU32(sensor(ADL_PMLOG_CLK_MEMCLK));
    telemetry.utilization = castU32(sensor(ADL_PMLOG_INFO_ACTIVITY_GFX));

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
    telemetry.power = castDouble(power);

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

    telemetry.coreClock = castU32(activity.iEngineClock) / 100u; // 10 kHz -> MHz
    telemetry.memoryClock = castU32(activity.iMemoryClock) / 100u;
    telemetry.utilization = castU32(activity.iActivityPercent);

    return telemetry;
}

#endif
