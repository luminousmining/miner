#define ALGO_ETHASH

#include <cuda.h>
#include <cuda_runtime.h>

#include <algo/dag_context.hpp>
#include <common/error/cuda_error.hpp>
#include <resolver/nvidia/ethash_kernel_parameter.hpp>

#include <common/cuda/be_u64.cuh>
#include <common/cuda/register.cuh>

#include <algo/crypto/cuda/fnv1.cuh>
#include <algo/crypto/cuda/keccak_f1600.cuh>

#include <algo/ethash/cuda/constants.cuh>
#include <algo/ethash/cuda/memory.cuh>
#include <algo/ethash/cuda/dag.cuh>
#include <algo/ethash/cuda/search.cuh>
