#pragma once

#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <common/kernel_generator/kernel_generator.hpp>


namespace common
{
    struct KernelGeneratorCuda : common::KernelGenerator
    {
    public:
        CUfunction cuFunction{ nullptr };

        bool build(uint32_t const deviceId,
                   uint32_t const major,
                   uint32_t const minor);
        bool occupancy(CUdevice* const cuDevice,
                       uint32_t const threads,
                       uint32_t const blocks);
    private:
        nvrtcProgram cuProgram{};
    };
}
#endif
