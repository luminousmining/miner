#pragma once

#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <resolver/nvidia/autolykos_v2_kernel_parameter.hpp>


bool autolykosv2FreeMemory(resolver::nvidia::autolykos_v2::KernelParameters& params);
bool autolykosv2InitMemory(resolver::nvidia::autolykos_v2::KernelParameters& params);
bool autolykosv2UpateConstants(resolver::nvidia::autolykos_v2::KernelParameters& params);
bool autolykosv2BuildDag(cudaStream_t stream,
                         resolver::nvidia::autolykos_v2::KernelParameters& params);
bool autolykosv2Search(cudaStream_t stream,
                       uint32_t const blocks,
                       uint32_t const threads,
                       resolver::nvidia::autolykos_v2::KernelParameters& params);

#endif
