#pragma once

#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <algo/blake3/result.hpp>
#include <algo/hash.hpp>
#include <common/opencl/buffer_mapped.hpp>


namespace resolver
{
    namespace amd
    {
        namespace blake3
        {
            // Blake3 (Alephium) is not memory-hard: per-job state is the 384-byte
            // header blob (nonce written per work-item) and the 32-byte target. No DAG.
            struct KernelParameters
            {
                common::opencl::BufferMapped<algo::hash3072> headerCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                                                                          | CL_MEM_ALLOC_HOST_PTR };
                common::opencl::BufferMapped<algo::hash256>  targetCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                                                                         | CL_MEM_ALLOC_HOST_PTR };
                common::opencl::BufferMapped<algo::blake3::Result> resultCache{ CL_MEM_READ_WRITE
                                                                                | CL_MEM_ALLOC_HOST_PTR };
            };
        }
    }
}

#endif
