#pragma once

#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <common/kernel_generator/kernel_generator.hpp>


namespace common
{
    struct KernelGeneratorOpenCL : common::KernelGenerator
    {
      public:
        cl::Kernel  clKernel{};
        // Public so a single built program can yield more than one kernel by name
        // (the OpenCL KATs pull test_hash/search/etc. from one build). Production
        // code only reads clKernel; widening visibility changes nothing for it.
        cl::Program clProgram{};

        void clear() final;
        bool build(cl::Device* const clDevice, cl::Context* const clContext);
    };
}
#endif
