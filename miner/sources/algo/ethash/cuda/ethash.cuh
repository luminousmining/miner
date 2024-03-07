#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <algo/dag_context.hpp>
#include <algo/ethash/result.hpp>
#include <resolver/nvidia/ethash_kernel_parameter.hpp>


bool ethashFreeMemory(resolver::nvidia::ethash::KernelParameters& params);
bool ethashInitMemory(algo::DagContext const& context,
                      resolver::nvidia::ethash::KernelParameters& params);
bool ethashUpdateConstants(uint32_t const* const header,
                           uint64_t const boundary);
bool ethashBuildDag(cudaStream_t stream,
                    uint32_t const dagItemParents,
                    uint32_t const dagNumberItems);
bool ethashSearch(cudaStream_t stream,
                  algo::ethash::Result* const result,
                  uint32_t const blocks,
                  uint32_t const threads,
                  uint64_t const startNonce);
