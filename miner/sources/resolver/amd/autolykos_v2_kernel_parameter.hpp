#pragma once

#include <CL/opencl.hpp>

#include <algo/autolykos/result.hpp>
#include <common/opencl/buffer_mapped.hpp>
#include <common/opencl/buffer_wrapper.hpp>

#define NEW_BUFFER

namespace resolver
{
    namespace amd
    {
        namespace autolykos_v2
        {
            struct KernelParameters
            {
                uint32_t hostPeriod { 0u };
                uint32_t hostHeight { 0u };
                uint32_t hostDagItemCount { 0u };
                uint64_t hostNonce { 0ull };
                algo::hash256 hostHeader {};

                common::opencl::Buffer<algo::u_hash256> BHashes { CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS };
                common::opencl::Buffer<algo::u_hash256> dagCache { CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS };
                common::opencl::BufferMapped<uint32_t> boundaryCache
                {
                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                    algo::LEN_HASH_256
                };
                common::opencl::BufferMapped<uint32_t> headerCache
                {
                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                    algo::LEN_HASH_256
                };
                common::opencl::BufferMapped<algo::autolykos_v2::Result> resultCache
                {
                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR
                };
            };
        }
    }
}