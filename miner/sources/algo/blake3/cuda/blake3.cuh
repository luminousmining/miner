#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <resolver/nvidia/blake3_kernel_parameter.hpp>


bool blake3FreeMemory(resolver::nvidia::blake3::KernelParameters& params);
bool blake3InitMemory(resolver::nvidia::blake3::KernelParameters& params);
bool blake3UpateConstants(resolver::nvidia::blake3::KernelParameters& params);
bool blake3Search(cudaStream_t stream,
                  resolver::nvidia::blake3::KernelParameters& params,
                  uint32_t const blocks,
                  uint32_t const threads);
