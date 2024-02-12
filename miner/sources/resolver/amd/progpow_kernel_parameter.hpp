#pragma once

#include <CL/opencl.hpp>

#include <algo/progpow/result.hpp>
#include <common/opencl/buffer_mapped.hpp>


namespace resolver
{
    namespace amd
    {
        namespace progpow
        {
            struct KernelParameters
            {
                cl::Buffer* lightCache { nullptr };
                cl::Buffer* dagCache { nullptr };
                common::opencl::BufferMapped<uint32_t> headerCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                                                    algo::LEN_HASH_256 };
                common::opencl::BufferMapped<algo::progpow::Result> resultCache{ CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR };
            };
        }
    }
}
