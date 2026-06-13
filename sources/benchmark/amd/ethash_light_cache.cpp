#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <algo/ethash/ethash.hpp>
#include <algo/hash.hpp>
#include <benchmark/workflow.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <common/opencl/buffer_wrapper.hpp>


bool benchmark::BenchmarkWorkflow::runAmdEthashLightCache()
{
    ////////////////////////////////////////////////////////////////////////////
    if (false == config.amd.isAlgoEnabled("ethash_light_cache"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark AMD Light Cache";

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard            dashboard{ createNewDashboard("[AMD] Light Cache") };
    benchmark::AlgoConfig const& algo{ config.amd.getAlgo("ethash_light_cache") };

    ////////////////////////////////////////////////////////////////////////////
    // Epoch-representative light-cache dimensions, matching the NVIDIA baseline
    // (1409017 items * 64 bytes = 90177088 bytes). The kernel runs the full
    // keccak chain and RandMemoHash rounds regardless of the buffer contents,
    // so an uninitialised buffer measures the same work as a real build.
    uint32_t const lightCacheNumber{ 1409017u };
    size_t const   lightCacheSize{ castSize(lightCacheNumber) * algo::LEN_HASH_512 };

    ////////////////////////////////////////////////////////////////////////////
    common::opencl::Buffer<algo::hash512> lightCache{ CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS };
    lightCache.setSize(lightCacheSize);
    if (false == lightCache.alloc(propertiesAmd.clContext))
    {
        logErr() << "Fail to allocate light cache buffer";
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    auto benchLightCache = [&](std::string const& kernelName, uint32_t const loop) -> bool
    {
        ///////////////////////////////////////////////////////////////////////
        common::KernelGeneratorOpenCL generator{};

        ///////////////////////////////////////////////////////////////////////
        generator.setKernelName(kernelName);

        ///////////////////////////////////////////////////////////////////////
        generator.addDefine("LIGHT_CACHE_ROUNDS", castU32(algo::ethash::LIGHT_CACHE_ROUNDS));

        ///////////////////////////////////////////////////////////////////////
        generator.appendFile("kernel/ethash_light_cache/" + kernelName + ".cl");

        ///////////////////////////////////////////////////////////////////////
        if (false == generator.build(&propertiesAmd.clDevice, &propertiesAmd.clContext))
        {
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        // Single work-item, single group: the light-cache build is strictly
        // sequential (keccak chain + i-1 read-after-write) and cannot be split.
        auto& clKernel{ generator.clKernel };
        OPENCL_ER(clKernel.setArg(0u, *lightCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(1u, lightCacheNumber));

        ///////////////////////////////////////////////////////////////////////
        setGrid(lightCacheNumber, 1u);

        ///////////////////////////////////////////////////////////////////////
        for (uint32_t i{ 0u }; i < loop; ++i)
        {
            startChrono(kernelName);
            OPENCL_ER(propertiesAmd.clQueue
                          .enqueueNDRangeKernel(clKernel, cl::NullRange, cl::NDRange(1, 1, 1), cl::NDRange(1, 1, 1)));
            OPENCL_ER(propertiesAmd.clQueue.finish());
            stopChrono(dashboard);
        }

        return true;
    };

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("ethash_light_cache_lm0"))
    {
        uint32_t const loop{ algo.resolveKernel("ethash_light_cache_lm0").loop };
        benchLightCache("ethash_light_cache_lm0", loop);
    }

    ////////////////////////////////////////////////////////////////////////////
    lightCache.free();

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}

#endif
