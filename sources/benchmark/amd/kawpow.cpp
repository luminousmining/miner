#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/progpow/progpow.hpp>
#include <benchmark/workflow.hpp>
#include <common/opencl/buffer_wrapper.hpp>
#include <common/opencl/buffer_mapped.hpp>
#include <common/custom.hpp>
#include <common/kernel_generator/opencl.hpp>


bool benchmark::BenchmarkWorkflow::runAmdKawpow()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    bool dagInitialized{ false };
    uint64_t const dagItems{ 16777213ull };
    uint64_t const dagItemsKawpow{ dagItems / 2ull };
    auto const header
    {
        algo::toHash256("71c967486cb3b70d5dfcb2ebd8eeef138453637cacbf3ccb580a41a7e96986bb")
    };

    ////////////////////////////////////////////////////////////////////////////
    common::opencl::Buffer<algo::hash1024> dagCache{ CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS };
    common::opencl::BufferMapped<uint32_t> headerCache
    {
        CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
        algo::LEN_HASH_256
    };
    common::opencl::BufferMapped<t_result> resultCache{ CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR };

    ////////////////////////////////////////////////////////////////////////////
    dagCache.setSize(dagItems * algo::LEN_HASH_1024);
    dagCache.alloc(propertiesAmd.clContext);
    headerCache.alloc(&propertiesAmd.clQueue, propertiesAmd.clContext);
    resultCache.alloc(&propertiesAmd.clQueue, propertiesAmd.clContext);

    ////////////////////////////////////////////////////////////////////////////
    if (false == headerCache.setBufferDevice(&propertiesAmd.clQueue,
                                             (uint32_t*)header.word32))
    {
        logErr() << "Fail to copy header in cache";
    }

    ///////////////////////////////////////////////////////////////////////////
    // Build kernel init_array
    // Initialize dagCache to simulate DAG for KAWPOW
    {
        ///////////////////////////////////////////////////////////////////////
        common::KernelGeneratorOpenCL generator{};

        ///////////////////////////////////////////////////////////////////////
        generator.setKernelName("init_array");
        generator.appendFile("kernel/common/init_array.cl");

        ///////////////////////////////////////////////////////////////////////
        if (true == generator.build(&propertiesAmd.clDevice,
                                    &propertiesAmd.clContext))
        {
            ///////////////////////////////////////////////////////////////////
            auto& clKernel{ generator.clKernel };
            OPENCL_ER(clKernel.setArg(0u, *dagCache.getBuffer()));
            OPENCL_ER(clKernel.setArg(1u, castU32(dagItems)));

            ///////////////////////////////////////////////////////////////////
            OPENCL_ER(
                propertiesAmd.clQueue.enqueueNDRangeKernel
                (
                    clKernel,
                    cl::NullRange,
                    cl::NDRange(1, 1, 1),
                    cl::NDRange(1, 1, 1)
                )
            );
            OPENCL_ER(propertiesAmd.clQueue.finish());

            ///////////////////////////////////////////////////////////////////
            dagInitialized = true;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    auto benchKawpow = [&](std::string const& kernelName,
                           uint32_t const groupSize,
                           uint32_t const workerGroupCount,
                           uint32_t const workItemCollaborate,
                           uint32_t const loop) -> bool
    {
        ///////////////////////////////////////////////////////////////////////
        common::KernelGeneratorOpenCL generator{};

        ///////////////////////////////////////////////////////////////////////
        uint32_t const batchGroupLane{ groupSize / workItemCollaborate };

        ///////////////////////////////////////////////////////////////////////
        generator.setKernelName(kernelName);

        ///////////////////////////////////////////////////////////////////////
        generator.addDefine("OCL_DIM", 2);
        generator.addDefine("GROUP_SIZE", groupSize);
        generator.addDefine("REGS", algo::progpow::REGS);
        generator.addDefine("STATE_SIZE", 25u);
        generator.addDefine("WAVEFRONT", propertiesAmd.clDevice.getInfo<CL_DEVICE_WAVEFRONT_WIDTH_AMD>());
        generator.addDefine("DIGEST_SIZE", 16u);
        generator.addDefine("HASH_SIZE", 32u);
        generator.addDefine("WORK_ITEM_COLLABORATE", workItemCollaborate);
        generator.addDefine("MODULE_CACHE", algo::progpow::MODULE_CACHE);
        generator.addDefine("HEADER_ITEM_BY_THREAD", algo::progpow::MODULE_CACHE / groupSize);
        generator.addDefine("COUNT_DAG", algo::progpow::COUNT_DAG);
        generator.addDefine("DAG_SIZE", castU32(dagItemsKawpow));
        generator.addDefine("BATCH_GROUP_LANE", batchGroupLane);
        generator.addDefine("SHARE_SEED_SIZE", batchGroupLane);
        generator.addDefine("SHARE_HASH0_SIZE", batchGroupLane);
        generator.addDefine("SHARE_FNV1A_SIZE", groupSize);
        generator.addDefine("MODULE_CACHE_GROUP", groupSize * 4u);
        generator.addDefine("MODULE_LOOP", algo::progpow::MODULE_CACHE / (groupSize / 4u));

        ///////////////////////////////////////////////////////////////////////
        generator.addInclude("kernel/common/debug.cl");
        generator.addInclude("kernel/common/grid.cl");
        generator.addInclude("kernel/common/cross_lane.cl");
        generator.addInclude("kernel/common/rotate_byte.cl");
        generator.addInclude("kernel/crypto/fnv1.cl");
        generator.addInclude("kernel/crypto/keccak_f800.cl");
        generator.addInclude("kernel/crypto/kiss99.cl");
        generator.addInclude("kernel/kawpow/sequence_dynamic.cl");
        generator.addInclude("kernel/kawpow/sequence_dynamic_local.cl");

        ///////////////////////////////////////////////////////////////////////
        generator.appendFile("kernel/common/result.cl");
        generator.appendFile("kernel/kawpow/" + kernelName + ".cl");

        ///////////////////////////////////////////////////////////////////////
        if (false == generator.build(&propertiesAmd.clDevice,
                                     &propertiesAmd.clContext))
        {
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        auto& clKernel{ generator.clKernel };
        OPENCL_ER(clKernel.setArg(0u, 0ull));
        OPENCL_ER(clKernel.setArg(1u, *headerCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(2u, *dagCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(3u, *resultCache.getBuffer()));

        ///////////////////////////////////////////////////////////////////////
        setGrid(groupSize, workerGroupCount);

        ///////////////////////////////////////////////////////////////////////
        for (uint32_t i{ 0u }; i < loop; ++i)
        {
            startChrono(kernelName);
            OPENCL_ER
            (
                propertiesAmd.clQueue.enqueueNDRangeKernel
                (
                    clKernel,
                    cl::NullRange,
                    cl::NDRange(groupSize, workerGroupCount, 1),
                    cl::NDRange(groupSize, 1, 1)
                )
            );
            OPENCL_ER(propertiesAmd.clQueue.finish());
            stopChrono(i);
        }

        return true;
    };

    ////////////////////////////////////////////////////////////////////////////
    if (true == dagInitialized)
    {
        benchKawpow("kawpow_lm1", 256u, 1024u, algo::progpow::LANES, 1u); // Parallele + LDS
        benchKawpow("kawpow_lm2", 256u, 1024u, algo::progpow::LANES, 1u); // Parallel + crosslane
        benchKawpow("kawpow_lm3", 256u, 1024u, algo::progpow::LANES, 1u); // Parallel + crosslane + LDS header
    }

    ////////////////////////////////////////////////////////////////////////////
    dagCache.free();
    headerCache.free();
    resultCache.free();

    ////////////////////////////////////////////////////////////////////////////
    return true;
}

#endif
