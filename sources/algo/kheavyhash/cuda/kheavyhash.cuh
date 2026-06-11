#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <resolver/nvidia/kheavyhash_kernel_parameter.hpp>


bool kheavyhashFreeMemory(resolver::nvidia::kheavyhash::KernelParameters& params);
bool kheavyhashInitMemory(resolver::nvidia::kheavyhash::KernelParameters& params);
bool kheavyhashUpdateConstants(resolver::nvidia::kheavyhash::KernelParameters& params);
void kheavyhashSearch(cudaStream_t stream,
                      resolver::nvidia::kheavyhash::KernelParameters& params,
                      uint32_t const currentIndexStream,
                      uint32_t const blocks,
                      uint32_t const threads);
