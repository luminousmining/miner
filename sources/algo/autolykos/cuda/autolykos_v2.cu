#define ALGO_AUTOLYKOS_V2

////////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
#include <algo/hash.hpp>
#include <algo/autolykos/autolykos.hpp>
#include <algo/autolykos/result.hpp>
#include <common/custom.hpp>
#include <common/error/cuda_error.hpp>
#include <resolver/nvidia/autolykos_v2_kernel_parameter.hpp>

////////////////////////////////////////////////////////////////////////////////
#include <common/cuda/be_u32.cuh>
#include <common/cuda/be_u64.cuh>
#include <common/cuda/rotate_byte.cuh>
#include <algo/autolykos/cuda/constants.cuh>
#include <algo/autolykos/cuda/memory.cuh>
#include <algo/crypto/cuda/blake2b.cuh>
#include <algo/autolykos/cuda/dag.cuh>
#include <algo/autolykos/cuda/search.cuh>
