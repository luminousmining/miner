#pragma once

#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <resolver/nvidia/random_x_kernel_parameter.hpp>


bool randomxFreeMemory(resolver::nvidia::random_x::KernelParameters& params);
bool randomxInitMemory(resolver::nvidia::random_x::KernelParameters& params,
                       uint32_t const blocks,
                       uint32_t const threads);
bool randomxBuildCache(cudaStream_t stream,
                       uint8_t* gpuCache,
                       uint8_t const* seedHash);
bool randomxBuildDataset(cudaStream_t stream,
                         resolver::nvidia::random_x::KernelParameters& params,
                         uint8_t const* gpuCache,
                         uint8_t const* seedHash);
bool randomxUpdateConstants(uint8_t const* const blob,
                            uint32_t const target,
                            uint64_t const startNonce);
bool randomxSearch(cudaStream_t stream,
                   uint32_t const blocks,
                   uint32_t const threads,
                   resolver::nvidia::random_x::KernelParameters& params);

#endif
