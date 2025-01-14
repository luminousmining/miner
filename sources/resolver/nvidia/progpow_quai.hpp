#pragma once

#if defined(CUDA_ENABLE)

#include <resolver/nvidia/progpow.hpp>


namespace resolver
{
    class ResolverNvidiaProgpowQuai : public resolver::ResolverNvidiaProgPOW
    {
    public:
        ResolverNvidiaProgpowQuai();
        ~ResolverNvidiaProgpowQuai() = default;
    };
}

#endif
