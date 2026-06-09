#pragma once

#if defined(AMD_ENABLE)

#include <cstdint>
#include <map>
#include <set>

#include <profiler/adl/adl_sdk.h>


namespace profiler
{
    struct Amd
    {
      public:
        ////////////////////////////////////////////////////////////////////////
        /// Backend-agnostic telemetry snapshot for a single GPU.
        struct Telemetry
        {
            uint32_t coreClock{ 0u };   // Engine/GFX clock (MHz)
            uint32_t memoryClock{ 0u }; // Memory clock (MHz)
            uint32_t utilization{ 0u }; // GPU activity (%)
            double   power{ 0.0 };      // Board/ASIC power draw (W)
        };

        bool valid{ false };

        bool      load();
        void      unload();
        Telemetry getTelemetry(uint32_t const pciBus);

      private:
        ////////////////////////////////////////////////////////////////////////
        /// Which AMD telemetry API the loaded driver actually supports. This is
        /// the single source of truth for the selected ADL version: PMLOG is the
        /// modern ADL2 path (RDNA/Vega+); OVERDRIVE5 is the legacy ADL1 fallback
        /// for pre-GCN1.2 cards. Once `load()` picks a backend, every call site
        /// dispatches on this enum instead of re-checking raw function pointers.
        /// A future ADLX backend slots in here.
        enum class Backend : uint8_t
        {
            NONE,
            PMLOG,     // ADL2 context + PMLog sensor query
            OVERDRIVE5 // ADL1 context-less Overdrive5 activity
        };

#if defined(_WIN32)
        HMODULE libModule{ nullptr };
#else
        void* libModule{ nullptr };
#endif

        ADL_CONTEXT_HANDLE      context{ nullptr };
        Backend                 backend{ Backend::NONE };
        std::map<uint32_t, int> busToAdapter{};   // PCI bus number -> ADL adapter index
        std::set<uint32_t>      warnedBuses{};     // Buses already warned about (avoid per-interval spam)
        std::set<int>           warnedAdapters{};  // Adapters whose PMLog query failed (warn once)

        ////////////////////////////////////////////////////////////////////////
        // ADL1 (legacy, context-less)
        using ADL_MAIN_CONTROL_CREATE        = int (*)(ADL_MAIN_MALLOC_CALLBACK callback, int enumConnectedAdapters);
        using ADL_MAIN_CONTROL_DESTROY       = int (*)();
        using ADL_ADAPTER_NUMBEROFADAPTERS_GET = int (*)(int* lpNumAdapters);
        using ADL_ADAPTER_ADAPTERINFO_GET    = int (*)(LPAdapterInfo lpInfo, int iInputSize);
        using ADL_PM_ACTIVITY_GET            = int (*)(int iAdapterIndex, ADLPMActivity* lpActivity);

        ////////////////////////////////////////////////////////////////////////
        // ADL2 (modern, context-based)
        using ADL2_MAIN_CONTROL_CREATE =
            int (*)(ADL_MAIN_MALLOC_CALLBACK callback, int enumConnectedAdapters, ADL_CONTEXT_HANDLE* context);
        using ADL2_MAIN_CONTROL_DESTROY         = int (*)(ADL_CONTEXT_HANDLE context);
        using ADL2_ADAPTER_NUMBEROFADAPTERS_GET = int (*)(ADL_CONTEXT_HANDLE context, int* lpNumAdapters);
        using ADL2_ADAPTER_ADAPTERINFO_GET = int (*)(ADL_CONTEXT_HANDLE context, LPAdapterInfo lpInfo, int iInputSize);
        using ADL2_PMLOG_QUERY_GET =
            int (*)(ADL_CONTEXT_HANDLE context, int iAdapterIndex, ADLPMLogDataOutput* lpDataOutput);

        ADL_MAIN_CONTROL_CREATE          adlMainControlCreate{ nullptr };
        ADL_MAIN_CONTROL_DESTROY         adlMainControlDestroy{ nullptr };
        ADL_ADAPTER_NUMBEROFADAPTERS_GET adlAdapterNumberOfAdaptersGet{ nullptr };
        ADL_ADAPTER_ADAPTERINFO_GET      adlAdapterAdapterInfoGet{ nullptr };
        ADL_PM_ACTIVITY_GET              adlOverdrive5CurrentActivityGet{ nullptr };

        ADL2_MAIN_CONTROL_CREATE          adl2MainControlCreate{ nullptr };
        ADL2_MAIN_CONTROL_DESTROY         adl2MainControlDestroy{ nullptr };
        ADL2_ADAPTER_NUMBEROFADAPTERS_GET adl2AdapterNumberOfAdaptersGet{ nullptr };
        ADL2_ADAPTER_ADAPTERINFO_GET      adl2AdapterAdapterInfoGet{ nullptr };
        ADL2_PMLOG_QUERY_GET              adl2NewQueryPMLogDataGet{ nullptr };

        void* loadFunction(char const* name, bool const required = true);
        bool  loadAdl2Symbols();
        bool  loadAdl1Symbols();
        bool  buildAdapterMap();
        bool  resolveAdapterIndex(uint32_t const pciBus, int& adapterIndex);

        Telemetry getTelemetryPMLog(int const adapterIndex);
        Telemetry getTelemetryOverdrive5(int const adapterIndex);
    };
}

#endif //! AMD_ENABLE
