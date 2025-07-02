#pragma once

#include <resolver/cpu/progpow.hpp>


namespace resolver
{
    class ResolverCpuKawPOW : public resolver::ResolverCpuProgPOW
    {
    public:
        ResolverCpuKawPOW();
        ~ResolverCpuKawPOW() = default;
    };
}
