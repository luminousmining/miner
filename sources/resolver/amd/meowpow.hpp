#pragma once

#if defined(AMD_ENABLE)

#include <resolver/amd/progpow.hpp>


namespace resolver
{
    class ResolverAmdMeowPOW : public resolver::ResolverAmdProgPOW
    {
    public:
        ResolverAmdMeowPOW();
        ~ResolverAmdMeowPOW() = default;
    };
}

#endif
