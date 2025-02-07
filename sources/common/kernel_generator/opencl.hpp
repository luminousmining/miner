#pragma once

#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <common/kernel_generator/kernel_generator.hpp>


namespace common
{
    struct KernelGeneratorOpenCL : common::KernelGenerator
    {
    public:
        cl::Kernel clKernel{};

        void clear() final;
        bool build(cl::Device* const clDevice,
                   cl::Context* const clContext);
    private:
        cl::Program clProgram{};
    };
}
#endif
