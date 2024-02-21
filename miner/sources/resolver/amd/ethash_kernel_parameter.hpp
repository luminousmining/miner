#pragma once

#include <CL/opencl.hpp>

#include <algo/ethash/result.hpp>
#include <common/opencl/buffer_mapped.hpp>
#include <common/opencl/buffer_wrapper.hpp>

#define NEW_BUFFER

namespace resolver
{
    namespace amd
    {
        namespace ethash
        {
            struct KernelParameters
            {
                common::opencl::Buffer<algo::u_hash512> lightCache { CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY };
                common::opencl::Buffer<algo::u_hash1024> dagCache { CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS };
                common::opencl::BufferMapped<uint32_t> headerCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                                                    algo::LEN_HASH_256 };
                common::opencl::BufferMapped<algo::ethash::Result> resultCache{ CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR };
            };
        }
    }
}