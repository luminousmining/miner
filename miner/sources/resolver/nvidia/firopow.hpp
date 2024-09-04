#pragma once

#if defined(CUDA_ENABLE)

#include <resolver/nvidia/progpow.hpp>


namespace resolver
{
    class ResolverNvidiaFiroPOW : public resolver::ResolverNvidiaProgPOW
    {
    public:
        ResolverNvidiaFiroPOW();
        ~ResolverNvidiaFiroPOW() = default;
    };
}

#endif
