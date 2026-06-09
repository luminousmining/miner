#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <algo/blake3/blake3.hpp>
#include <algo/blake3/result.hpp>
#include <algo/hash.hpp>
#include <benchmark/workflow.hpp>
#include <common/cast.hpp>
#include <common/error/opencl_error.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <common/opencl/buffer_mapped.hpp>


bool benchmark::BenchmarkWorkflow::runAmdBlake3()
{
    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark AMD Blake3";

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.amd.isAlgoEnabled("blake3"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard            dashboard{ createNewDashboard("[AMD] BLAKE3") };
    benchmark::AlgoConfig const& algo{ config.amd.getAlgo("blake3") };

    ////////////////////////////////////////////////////////////////////////////
    // No DAG. Buffer types mirror resolver/amd/blake3_kernel_parameter.hpp.
    common::opencl::BufferMapped<algo::hash3072>       headerCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                                                              | CL_MEM_ALLOC_HOST_PTR };
    common::opencl::BufferMapped<algo::hash256>        targetCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                                                             | CL_MEM_ALLOC_HOST_PTR };
    common::opencl::BufferMapped<algo::blake3::Result> resultCache{ CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR };

    ////////////////////////////////////////////////////////////////////////////
    headerCache.alloc(&propertiesAmd.clQueue, propertiesAmd.clContext);
    targetCache.alloc(&propertiesAmd.clQueue, propertiesAmd.clContext);
    resultCache.alloc(&propertiesAmd.clQueue, propertiesAmd.clContext);

    ////////////////////////////////////////////////////////////////////////////
    // Throughput is data-independent: any fixed header works. Target all-zero so
    // isLowerOrEqual returns false at byte 0 for every thread -> no result writes,
    // isolating pure powHash throughput.
    algo::hash3072 header{};
    for (uint32_t i{ 0u }; i < algo::blake3::HEADER_U32_CAP; ++i)
    {
        header.word32[i] = 0x01020304u + i;
    }
    algo::hash256 target{};

    if (false == headerCache.setBufferDevice(&propertiesAmd.clQueue, &header))
    {
        logErr() << "Fail to copy header in cache";
    }
    if (false == targetCache.setBufferDevice(&propertiesAmd.clQueue, &target))
    {
        logErr() << "Fail to copy target in cache";
    }

    ////////////////////////////////////////////////////////////////////////////
    auto benchBlake3 =
        [&](std::string const& variant, uint32_t const loop, uint32_t const threads, uint32_t const blocks) -> bool
    {
        ////////////////////////////////////////////////////////////////////////
        common::KernelGeneratorOpenCL generator{};
        generator.setKernelName("search");
        generator.addDefine("MAX_RESULT", algo::blake3::MAX_RESULT);

        ////////////////////////////////////////////////////////////////////////
        // Production Alephium kernel; it #includes kernel/crypto/blake3.cl, which
        // the generator resolves.
        if (false == generator.appendFile("kernel/blake3/blake3.cl"))
        {
            return false;
        }
        if (false == generator.build(&propertiesAmd.clDevice, &propertiesAmd.clContext))
        {
            return false;
        }

        ////////////////////////////////////////////////////////////////////////
        auto& clKernel{ generator.clKernel };
        OPENCL_ER(clKernel.setArg(0u, *headerCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(1u, *targetCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(2u, 0ull));
        OPENCL_ER(clKernel.setArg(3u, 0u));
        OPENCL_ER(clKernel.setArg(4u, 0u));
        OPENCL_ER(clKernel.setArg(5u, *resultCache.getBuffer()));

        ////////////////////////////////////////////////////////////////////////
        setGrid(threads, blocks);

        ////////////////////////////////////////////////////////////////////////
        // Alephium search uses get_global_id(0): launch 1D (matches the resolver's
        // executeSync), NOT the 2D grid the kawpow bench uses.
        size_t const globalSize{ static_cast<size_t>(threads) * static_cast<size_t>(blocks) };
        size_t const localSize{ static_cast<size_t>(threads) };
        for (uint32_t i{ 0u }; i < loop; ++i)
        {
            startChrono(variant);
            OPENCL_ER(
                propertiesAmd.clQueue
                    .enqueueNDRangeKernel(clKernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize)));
            OPENCL_ER(propertiesAmd.clQueue.finish());
            stopChrono(dashboard);
        }

        return true;
    };

    ////////////////////////////////////////////////////////////////////////////
    auto const runKernel = [&](std::string const& name)
    {
        if (false == algo.isKernelEnabled(name))
        {
            return;
        }
        KernelParams const p{ algo.resolveKernel(name) };
        benchBlake3(name, p.loop, p.threads, p.blocks);
    };

    ////////////////////////////////////////////////////////////////////////////
    runKernel("blake3"); // Alephium production search kernel — throughput baseline

    ////////////////////////////////////////////////////////////////////////////
    headerCache.free();
    targetCache.free();
    resultCache.free();

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}

#endif
