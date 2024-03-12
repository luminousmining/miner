#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <device/nvidia.hpp>
#include <resolver/resolver.hpp>


namespace resolver
{
    struct ResolverNvidia : public resolver::Resolver
    {
    public:
        bool            isDoubleStream { false };
        cudaStream_t    cuStream[device::DeviceNvidia::MAX_INDEX_STREAM]{ nullptr, nullptr };
        cudaDeviceProp* cuProperties{ nullptr };

        virtual ~ResolverNvidia() = default;

        size_t       getCurrentIndex() const;
        size_t       getNextIndex() const;
        cudaStream_t getCurrentStream();
        cudaStream_t getNextStream();
        void         swapStream();

    protected:
        size_t currentIndexStream { 0u };
        size_t nextIndexStream { 1u };
    };
}
