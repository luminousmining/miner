#pragma once

#if defined(AMD_ENABLE)

#include <resolver/amd/progpow.hpp>


namespace resolver
{
    class ResolverAmdProgpowQuai : public resolver::ResolverAmdProgPOW
    {
    public:
        ResolverAmdProgpowQuai();
        ~ResolverAmdProgpowQuai() = default;
    };
}

#endif
