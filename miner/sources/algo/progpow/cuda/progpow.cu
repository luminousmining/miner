#define ALGO_PROGPOW

#include <cuda.h>
#include <cuda_runtime.h>

#include <algo/dag_context.hpp>
#include <common/error/cuda_error.hpp>
#include <resolver/nvidia/progpow_kernel_parameter.hpp>

#include <common/cuda/be_u64.cuh>
#include <common/cuda/copy_u4.cuh>

#include <algo/crypto/cuda/fnv1.cuh>
#include <algo/crypto/cuda/keccak_f800.cuh>
#include <algo/crypto/cuda/keccak_f1600.cuh>

#include <algo/progpow/cuda/constants.cuh>
#include <algo/progpow/cuda/memory.cuh>
#include <algo/progpow/cuda/dag.cuh>
