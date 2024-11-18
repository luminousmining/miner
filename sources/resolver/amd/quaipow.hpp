#pragma once

#if defined(AMD_ENABLE)

#include <resolver/amd/progpow.hpp>


namespace resolver
{
    class ResolverAmdQuaiPOW : public resolver::ResolverAmdProgPOW
    {
    public:
        ResolverAmdQuaiPOW();
        ~ResolverAmdQuaiPOW() = default;
    };
}

#endif
