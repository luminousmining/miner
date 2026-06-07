#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <algo/dag_context.hpp>
#include <algo/ethash/ethash.hpp>
#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <benchmark/workflow.hpp>
#include <common/custom.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <common/opencl/buffer_mapped.hpp>
#include <common/opencl/buffer_wrapper.hpp>


bool benchmark::BenchmarkWorkflow::runAmdEthash()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark AMD Ethash";

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.amd.isAlgoEnabled("ethash"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard            dashboard{ createNewDashboard("[AMD] ETHASH") };
    benchmark::AlgoConfig const& algo{ config.amd.getAlgo("ethash") };

    ////////////////////////////////////////////////////////////////////////////
    bool                dagInitialized{ false };
    algo::hash256 const headerHash{ algo::toHash256(
        "71c967486cb3b70d5dfcb2ebd8eeef138453637cacbf3ccb580a41a7e96986bb") };
    algo::hash256 const seedHash{ algo::toHash256("7c4fb8a5d141973b69b521ce76b0dc50f0d2834d817c7f8310a6ab5becc6bb0c") };
    int32_t const epoch{ algo::ethash::ContextGenerator::instance().findEpoch(seedHash, algo::ethash::EPOCH_LENGTH) };

    ////////////////////////////////////////////////////////////////////////////
    common::opencl::Buffer<algo::hash512>  lightCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY };
    common::opencl::Buffer<algo::hash1024> dagCache{ CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS };
    common::opencl::BufferMapped<uint32_t> headerCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                                                            | CL_MEM_ALLOC_HOST_PTR,
                                                        algo::LEN_HASH_256 };
    common::opencl::BufferMapped<t_result> resultCache{ CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR };

    ////////////////////////////////////////////////////////////////////////////
    algo::DagContext dagContext{};
    algo::ethash::ContextGenerator::instance().build(
        algo::ALGORITHM::ETHASH,
        dagContext,
        epoch,
        algo::ethash::MAX_EPOCH_NUMBER,
        algo::ethash::DAG_COUNT_ITEMS_GROWTH,
        algo::ethash::DAG_COUNT_ITEMS_INIT,
        algo::ethash::LIGHT_CACHE_COUNT_ITEMS_GROWTH,
        algo::ethash::LIGHT_CACHE_COUNT_ITEMS_INIT,
        true /*config.deviceAlgorithm.ethashBuildLightCacheCPU*/
    );

    ////////////////////////////////////////////////////////////////////////////
    dagCache.setSize(dagContext.dagCache.size);
    lightCache.setSize(dagContext.lightCache.size);

    ////////////////////////////////////////////////////////////////////////////
    dagCache.alloc(propertiesAmd.clContext);
    lightCache.alloc(propertiesAmd.clContext);
    headerCache.alloc(&propertiesAmd.clQueue, propertiesAmd.clContext);
    resultCache.alloc(&propertiesAmd.clQueue, propertiesAmd.clContext);

    ////////////////////////////////////////////////////////////////////////////
    if (false == headerCache.setBufferDevice(&propertiesAmd.clQueue, (uint32_t*)headerHash.word32))
    {
        logErr() << "Fail to copy header in cache";
    }
    if (false == lightCache.write(dagContext.lightCache.hash, dagContext.lightCache.size, &propertiesAmd.clQueue))
    {
        logErr() << "Fail to copy light cache in cache";
    }

    ///////////////////////////////////////////////////////////////////////////
    // Build DAG (shared ethash_build_dag kernel)
    {
        ///////////////////////////////////////////////////////////////////////
        common::KernelGeneratorOpenCL generator{};

        ///////////////////////////////////////////////////////////////////////
        generator.setKernelName("ethash_build_dag");

        ///////////////////////////////////////////////////////////////////////
        generator.addDefine("GROUP_SIZE", 256u);
        generator.addDefine("DAG_LOOP", algo::ethash::DAG_ITEM_PARENTS / 4u / 4u);

        ///////////////////////////////////////////////////////////////////////
        generator.appendFile("kernel/ethash/ethash_dag.cl");

        ///////////////////////////////////////////////////////////////////////
        if (true == generator.build(&propertiesAmd.clDevice, &propertiesAmd.clContext))
        {
            auto& clKernel{ generator.clKernel };
            OPENCL_ER(clKernel.setArg(0u, *dagCache.getBuffer()));
            OPENCL_ER(clKernel.setArg(1u, *lightCache.getBuffer()));
            OPENCL_ER(clKernel.setArg(2u, algo::ethash::DAG_ITEM_PARENTS));
            OPENCL_ER(clKernel.setArg(3u, castU32(dagContext.dagCache.numberItem)));
            OPENCL_ER(clKernel.setArg(4u, castU32(dagContext.lightCache.numberItem)));

            uint32_t const maxGroupSize{ 256u };
            uint32_t const threadKernel{ castU32(dagContext.dagCache.numberItem) / maxGroupSize };
            OPENCL_ER(propertiesAmd.clQueue.enqueueNDRangeKernel(
                clKernel,
                cl::NullRange,
                cl::NDRange(maxGroupSize, threadKernel, 1),
                cl::NDRange(maxGroupSize, 1, 1)));
            OPENCL_ER(propertiesAmd.clQueue.finish());

            dagInitialized = true;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Each variant is a self-contained copy of the production ethash_search
    // kernel; the kernel function is always named "ethash_search" while the
    // file (and the dashboard label) carries the variant name. This lets us
    // A/B the barrier vs sub_group_barrier change against an identical harness.
    auto benchEthash = [&](std::string const& variantName,
                           uint32_t const     loop,
                           uint32_t const     groupSize,
                           uint32_t const     workerGroupCount) -> bool
    {
        ///////////////////////////////////////////////////////////////////////
        common::KernelGeneratorOpenCL generator{};

        ///////////////////////////////////////////////////////////////////////
        uint32_t const laneParallel{ 8u };
        uint32_t const groupParallel{ groupSize / laneParallel };
        uint32_t const lenSeed{ 4u };
        uint32_t const lenState{ 25u };

        ///////////////////////////////////////////////////////////////////////
        generator.setKernelName("ethash_search");

        ///////////////////////////////////////////////////////////////////////
        generator.addDefine("GROUP_SIZE", groupSize);
        generator.addDefine("DAG_NUMBER_ITEM", castU32(dagContext.dagCache.numberItem));
        generator.addDefine("LANE_PARALLEL", laneParallel);
        generator.addDefine("LEN_SEED", lenSeed);
        generator.addDefine("LEN_STATE", lenState);
        generator.addDefine("LEN_HASHES", groupParallel * lenSeed);
        generator.addDefine("LEN_WORD0", groupSize);
        generator.addDefine("LEN_REDUCE", groupSize);
        generator.addDefine("LEN_SWAPPER", groupParallel);
        generator.addDefine("LEN_KECCAK", 24u);
        generator.addDefine("MAX_KECCAK_ROUND", 23u);

        ///////////////////////////////////////////////////////////////////////
        generator.appendFile("kernel/ethash/" + variantName + ".cl");

        ///////////////////////////////////////////////////////////////////////
        if (false == generator.build(&propertiesAmd.clDevice, &propertiesAmd.clContext))
        {
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        auto& clKernel{ generator.clKernel };
        OPENCL_ER(clKernel.setArg(0u, *dagCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(1u, *resultCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(2u, *headerCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(3u, 0ull));
        OPENCL_ER(clKernel.setArg(4u, 0ull));

        ///////////////////////////////////////////////////////////////////////
        setGrid(groupSize, workerGroupCount);

        ///////////////////////////////////////////////////////////////////////
        for (uint32_t i{ 0u }; i < loop; ++i)
        {
            startChrono(variantName);
            OPENCL_ER(propertiesAmd.clQueue.enqueueNDRangeKernel(
                clKernel,
                cl::NullRange,
                cl::NDRange(groupSize, workerGroupCount, 1),
                cl::NDRange(groupSize, 1, 1)));
            OPENCL_ER(propertiesAmd.clQueue.finish());
            stopChrono(dashboard);
        }

        return true;
    };

    ////////////////////////////////////////////////////////////////////////////
    if (true == dagInitialized)
    {
        auto const runKernel = [&](std::string const& name)
        {
            if (false == algo.isKernelEnabled(name))
            {
                return;
            }
            KernelParams const p{ algo.resolveKernel(name) };
            benchEthash(name, p.loop, p.threads, p.blocks);
        };

        runKernel("ethash_search_baseline");
        runKernel("ethash_search_subgroup");
    }

    ////////////////////////////////////////////////////////////////////////////
    algo::ethash::ContextGenerator::instance().free(algo::ALGORITHM::ETHASH);

    ////////////////////////////////////////////////////////////////////////////
    dagCache.free();
    headerCache.free();
    resultCache.free();

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}

#endif
