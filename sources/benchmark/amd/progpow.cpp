#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <algo/dag_context.hpp>
#include <algo/ethash/ethash.hpp>
#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/progpow/kawpow.hpp>
#include <algo/progpow/progpow.hpp>
#include <benchmark/workflow.hpp>
#include <common/custom.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <common/opencl/buffer_mapped.hpp>
#include <common/opencl/buffer_wrapper.hpp>


////////////////////////////////////////////////////////////////////////////////
// A/B of the production KAWPOW progpow_search kernel: baseline (full-work-group
// barrier + strided AoS LDS store) vs this PR's change (coalesced LDS store +
// sub_group_barrier). Both variants are byte-for-byte copies of the production
// progpow.cl; everything else in the kernel (result struct, kawpow_functions,
// the generated per-period math sequence) is shared, so the delta is exactly
// the optimisation. Assembly mirrors ResolverAmdProgPOW::buildSearch.
////////////////////////////////////////////////////////////////////////////////


bool benchmark::BenchmarkWorkflow::runAmdProgpow()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark AMD ProgPOW (kawpow A/B)";

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.amd.isAlgoEnabled("progpow"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard            dashboard{ createNewDashboard("[AMD] PROGPOW") };
    benchmark::AlgoConfig const& algo{ config.amd.getAlgo("progpow") };

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
        algo::ALGORITHM::KAWPOW,
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
        generator.addDefine("DAG_LOOP", algo::kawpow::DAG_ITEM_PARENTS / 4u / 4u);

        ///////////////////////////////////////////////////////////////////////
        generator.appendFile("kernel/ethash/ethash_dag.cl");

        ///////////////////////////////////////////////////////////////////////
        if (true == generator.build(&propertiesAmd.clDevice, &propertiesAmd.clContext))
        {
            auto& clKernel{ generator.clKernel };
            OPENCL_ER(clKernel.setArg(0u, *dagCache.getBuffer()));
            OPENCL_ER(clKernel.setArg(1u, *lightCache.getBuffer()));
            OPENCL_ER(clKernel.setArg(2u, algo::kawpow::DAG_ITEM_PARENTS));
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
    // The per-period random-math sequence is generated once (KAWPOW, device 0,
    // period 0) and shared by both variants -> kernel/progpow/sequence_math_random_0_0.cl
    uint32_t const benchDeviceId{ 0u };
    uint64_t const benchPeriod{ 0ull };
    algo::progpow::writeMathRandomKernelOpenCL(
        algo::progpow::VERSION::KAWPOW,
        benchDeviceId,
        benchPeriod,
        algo::kawpow::COUNT_CACHE,
        algo::kawpow::COUNT_MATH,
        algo::progpow::REGS,
        algo::progpow::MODULE_SOURCE);
    std::string const fileSequenceMath{ "kernel/progpow/sequence_math_random_"s + std::to_string(benchDeviceId)
                                        + "_"s + std::to_string(benchPeriod) + ".cl"s };

    ///////////////////////////////////////////////////////////////////////////
    auto benchProgpow = [&](std::string const& variantName,
                            uint32_t const     loop,
                            uint32_t const     groupSize,
                            uint32_t const     workerGroupCount) -> bool
    {
        ///////////////////////////////////////////////////////////////////////
        common::KernelGeneratorOpenCL generator{};

        ///////////////////////////////////////////////////////////////////////
        uint32_t const batchGroupLane{ groupSize / algo::progpow::LANES };

        ///////////////////////////////////////////////////////////////////////
        generator.setKernelName("progpow_search");
        generator.declareDefine("__KERNEL_KAWPOW");

        ///////////////////////////////////////////////////////////////////////
        generator.addDefine("GROUP_SIZE", groupSize);
        generator.addDefine("MAX_RESULT", 4u);
        generator.addDefine("REGS", algo::progpow::REGS);
        generator.addDefine("LANES", algo::progpow::LANES);
        generator.addDefine("MODULE_CACHE", algo::progpow::MODULE_CACHE);
        generator.addDefine("COUNT_DAG", algo::progpow::COUNT_DAG);
        generator.addDefine("DAG_SIZE", castU32(dagContext.dagCache.numberItem / 2ull));
        generator.addDefine("BATCH_GROUP_LANE", batchGroupLane);
        generator.addDefine("SHARE_SEED_SIZE", batchGroupLane);
        generator.addDefine("SHARE_HASH0_SIZE", batchGroupLane);
        generator.addDefine("SHARE_FNV1A_SIZE", groupSize);
        generator.addDefine("MODULE_CACHE_GROUP", groupSize * 4u);
        generator.addDefine("MODULE_LOOP", algo::progpow::MODULE_CACHE / (groupSize / 4u));
        generator.addDefine("TOTAL_THREADS", groupSize * workerGroupCount);

        ///////////////////////////////////////////////////////////////////////
        generator.addInclude("kernel/common/rotate_byte.cl");
        generator.addInclude("kernel/crypto/fnv1.cl");
        generator.addInclude("kernel/crypto/keccak_f800.cl");
        generator.addInclude("kernel/crypto/kiss99.cl");

        ///////////////////////////////////////////////////////////////////////
        generator.appendLine("#pragma OPENCL EXTENSION cl_khr_fp16 : enable");

        ///////////////////////////////////////////////////////////////////////
        if (false == generator.appendFile("kernel/progpow/progpow_result.cl")
            || false == generator.appendFile("kernel/progpow/kawpow_functions.cl")
            || false == generator.appendFile(fileSequenceMath)
            || false == generator.appendFile("kernel/progpow/" + variantName + ".cl"))
        {
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        if (false == generator.build(&propertiesAmd.clDevice, &propertiesAmd.clContext))
        {
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        auto& clKernel{ generator.clKernel };
        OPENCL_ER(clKernel.setArg(0u, 0ull)); // nonce
        OPENCL_ER(clKernel.setArg(1u, 0ull)); // boundary
        OPENCL_ER(clKernel.setArg(2u, *headerCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(3u, *dagCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(4u, *resultCache.getBuffer()));

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
            benchProgpow(name, p.loop, p.threads, p.blocks);
        };

        runKernel("progpow_baseline");
        runKernel("progpow_subgroup");
    }

    ////////////////////////////////////////////////////////////////////////////
    algo::ethash::ContextGenerator::instance().free(algo::ALGORITHM::KAWPOW);

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
