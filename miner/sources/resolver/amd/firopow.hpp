#pragma once

#include <resolver/amd/progpow.hpp>


namespace resolver
{
    class ResolverAmdFiroPOW : public resolver::ResolverAmdProgPOW
    {
    public:
        ResolverAmdFiroPOW();
        ~ResolverAmdFiroPOW() = default;
    };
}
