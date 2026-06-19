#if defined(CPU_ENABLE)

#include <common/cast.hpp>
#include <common/log/log.hpp>
#include <resolver/cpu/cpu_affinity.hpp>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif


bool resolver::cpu::pinThisThreadToCore([[maybe_unused]] uint32_t const coreIndex)
{
#if defined(_WIN32)
    DWORD_PTR const mask{ castDWORDPTR(1ull << coreIndex) };
    bool const      success{ 0 != SetThreadAffinityMask(GetCurrentThread(), mask) };
    return success;
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(coreIndex, &set);
    bool const success{ 0 == pthread_setaffinity_np(pthread_self(), sizeof(set), &set) };
    return success;
#else
    logWarn() << "CPU affinity pinning is unavailable on this platform; --cpu_affinity ignored.";
    return false;
#endif
}

#endif
