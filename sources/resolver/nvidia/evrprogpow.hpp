#pragma once

#if defined(CUDA_ENABLE)

#include <resolver/nvidia/progpow.hpp>


namespace resolver
{
    class ResolverNvidiaEvrprogPOW : public resolver::ResolverNvidiaProgPOW
    {
    public:
        ResolverNvidiaEvrprogPOW();
        ~ResolverNvidiaEvrprogPOW() = default;
    };
}

#endif
