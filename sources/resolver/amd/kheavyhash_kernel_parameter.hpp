#pragma once

#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <algo/hash.hpp>
#include <algo/kheavyhash/result.hpp>
#include <common/opencl/buffer_mapped.hpp>
#include <common/opencl/buffer_wrapper.hpp>


namespace resolver
{
    namespace amd
    {
        namespace kheavyhash
        {
            // kHeavyHash is not memory-hard: per-job state is just the 64x64
            // nibble matrix (generated host-side and uploaded), the 32-byte
            // pre-pow header and the 32-byte little-endian target. No DAG.
            struct KernelParameters
            {
                common::opencl::Buffer<uint16_t> matrixCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                                              64u * 64u * sizeof(uint16_t) };
                common::opencl::BufferMapped<algo::hash256> headerCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                                                                         | CL_MEM_ALLOC_HOST_PTR };
                common::opencl::BufferMapped<algo::hash256> targetCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                                                                         | CL_MEM_ALLOC_HOST_PTR };
                common::opencl::BufferMapped<algo::kheavyhash::Result> resultCache{ CL_MEM_READ_WRITE
                                                                                    | CL_MEM_ALLOC_HOST_PTR };
            };
        }
    }
}

#endif
