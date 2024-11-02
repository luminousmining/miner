#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <common/error/cuda_error.hpp>


bool ethash_v0(cudaStream_t stream,
                uint32_t const blocks,
                uint32_t const threads);

