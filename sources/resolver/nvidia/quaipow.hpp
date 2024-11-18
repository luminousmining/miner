#pragma once

#if defined(CUDA_ENABLE)

#include <resolver/nvidia/progpow.hpp>


namespace resolver
{
    class ResolverNvidiaQuaiPOW : public resolver::ResolverNvidiaProgPOW
    {
    public:
        ResolverNvidiaQuaiPOW();
        ~ResolverNvidiaQuaiPOW() = default;
    };
}

#endif
