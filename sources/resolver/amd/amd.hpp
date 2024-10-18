#pragma once

#include <CL/opencl.hpp>

#include <resolver/resolver.hpp>


namespace resolver
{
    class ResolverAmd : public resolver::Resolver
    {
    public:

        virtual ~ResolverAmd() = default;

        void setDevice(cl::Device* const device);
        void setContext(cl::Context* const context);
        void setQueue(cl::CommandQueue* const queue);

        uint32_t getMaxGroupSize() const;

    protected:
        cl::Context*      clContext{ nullptr };
        cl::Device*       clDevice{ nullptr };
        cl::CommandQueue* clQueue{ nullptr };
    };
}
