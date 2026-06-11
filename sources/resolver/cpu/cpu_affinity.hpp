#pragma once

#include <cstdint>


namespace resolver
{
    // Pin the CALLING thread to a single logical core. Returns true on success.
    // macOS (and any platform without a hard-affinity API): best-effort no-op, returns false.
    bool pinThisThreadToCore(uint32_t coreIndex);
}
