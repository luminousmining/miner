#define ALGO_BLAKE3

////////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
#include <algo/hash.hpp>
#include <algo/blake3/blake3.hpp>
#include <algo/blake3/result.hpp>
#include <common/custom.hpp>
#include <common/error/cuda_error.hpp>
#include <resolver/nvidia/blake3_kernel_parameter.hpp>

////////////////////////////////////////////////////////////////////////////////
#include <common/cuda/debug.cuh>
#include <common/cuda/be_u64.cuh>
#include <common/cuda/compare.cuh>
#include <common/cuda/rotate_byte.cuh>
#include <algo/blake3/cuda/memory.cuh>
#include <algo/blake3/cuda/blake3_compress.cuh>
#include <algo/blake3/cuda/search.cuh>
