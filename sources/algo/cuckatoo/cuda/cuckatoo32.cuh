#pragma once

#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <resolver/nvidia/cuckatoo32_kernel_parameter.hpp>


////////////////////////////////////////////////////////////////////////////
// Host-callable functions that wrap the Cuckatoo32 CUDA kernels.
// These are implemented in cuckatoo32.cu and called from the resolver.
////////////////////////////////////////////////////////////////////////////

/// Allocate all GPU buffers listed in KernelParameters.
/// Returns false on allocation failure.
bool cuckatoo32AllocMemory(resolver::nvidia::cuckatoo32::KernelParameters& params);

/// Free all GPU buffers in KernelParameters.
bool cuckatoo32FreeMemory(resolver::nvidia::cuckatoo32::KernelParameters& params);

/// Upload the 4 SipHash keys (k0..k3) to GPU constant memory.
bool cuckatoo32UpdateConstants(resolver::nvidia::cuckatoo32::KernelParameters& params);

/// Run the lean-trimming pipeline (seed + TRIM_ROUNDS × count+trim).
/// Kernel grid size is DEFAULT_BLOCKS × DEFAULT_THREADS (hardcoded in .cu).
/// @param stream  CUDA stream to use
bool cuckatoo32Trim(
    cudaStream_t                                    stream,
    resolver::nvidia::cuckatoo32::KernelParameters& params);

/// Compact surviving edges and search for a 42-cycle on CPU.
/// @param params     KernelParameters with live devEdgeBitmap and SipHash keys
/// @param outFound   set to true when a cycle is found
/// @param outProof   42-element sorted proof written when outFound is true
bool cuckatoo32FindCycle(
    resolver::nvidia::cuckatoo32::KernelParameters& params,
    bool*                                           outFound,
    uint32_t                                        outProof[algo::cuckatoo::PROOF_SIZE]);

#endif // CUDA_ENABLE
