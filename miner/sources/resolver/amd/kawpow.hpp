#pragma once

#include <resolver/amd/progpow.hpp>


namespace resolver
{
    class ResolverAmdKawPOW : public resolver::ResolverAmdProgPOW
    {
    public:
        ResolverAmdKawPOW();
        ~ResolverAmdKawPOW() = default;
    };
}
