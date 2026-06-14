#if defined(CPU_ENABLE)

#include <common/log/log.hpp>
#include <resolver/cpu/cpu_affinity.hpp>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif


bool resolver::pinThisThreadToCore([[maybe_unused]] uint32_t const coreIndex)
{
#if defined(_WIN32)
    DWORD_PTR const mask{ static_cast<DWORD_PTR>(1ull << coreIndex) };
    return 0 != SetThreadAffinityMask(GetCurrentThread(), mask);
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(coreIndex, &set);
    return 0 == pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    static bool warned{ false };
    if (false == warned)
    {
        logWarn() << "CPU affinity pinning is unavailable on this platform; --cpu_affinity ignored.";
        warned = true;
    }
    return false;
#endif
}

#endif
