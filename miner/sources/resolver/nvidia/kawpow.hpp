#pragma once

#include <resolver/nvidia/progpow.hpp>


namespace resolver
{
    class ResolverNvidiaKawPOW : public resolver::ResolverNvidiaProgPOW
    {
    public:
        ResolverNvidiaKawPOW();
        ~ResolverNvidiaKawPOW() = default;
    };
}
