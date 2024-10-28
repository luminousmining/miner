#pragma once

#if defined(CUDA_ENABLE)
#include <cuda.h>
#include <cuda_runtime.h>

#include <resolver/resolver.hpp>


namespace resolver
{
    struct ResolverNvidia : public resolver::Resolver
    {
    public:
        cudaStream_t    cuStream{ nullptr };
        cudaDeviceProp* cuProperties{ nullptr };

        virtual ~ResolverNvidia() = default;

    protected:
        void overrideOccupancy(uint32_t const defaultThreads,
                               uint32_t const defaultBlocks) final;
    };
}

#endif
