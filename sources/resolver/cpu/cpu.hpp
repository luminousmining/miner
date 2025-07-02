#pragma once

#include <resolver/resolver.hpp>


namespace resolver
{
    struct ResolverCpu : public resolver::Resolver
    {
    public:
        virtual ~ResolverCpu() = default;

    protected:
        void overrideOccupancy(uint32_t const defaultThreads,
                               uint32_t const defaultBlocks) final;
    };
}
