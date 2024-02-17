#pragma once

#include <resolver/amd/progpow.hpp>


namespace resolver
{
    class ResolverAmdEvrprogPOW : public resolver::ResolverAmdProgPOW
    {
    public:
        ResolverAmdEvrprogPOW();
        ~ResolverAmdEvrprogPOW() = default;
    };
}
