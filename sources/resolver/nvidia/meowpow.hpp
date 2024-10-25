#pragma once

#if defined(CUDA_ENABLE)

#include <resolver/nvidia/progpow.hpp>


namespace resolver
{
    class ResolverNvidiaMeowPOW : public resolver::ResolverNvidiaProgPOW
    {
    public:
        ResolverNvidiaMeowPOW();
        ~ResolverNvidiaMeowPOW() = default;
    };
}

#endif
