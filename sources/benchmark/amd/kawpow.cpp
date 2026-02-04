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
    uint64_t const dagItems{ 16777213ull };
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
    if (false == headerCache.setBufferDevice(&propertiesAmd.clQueue, (uint32_t*)header.word32))
    {
        logErr() << "Fail to copy header in cache";
    }

    ////////////////////////////////////////////////////////////////////////////
    {
        common::KernelGeneratorOpenCL generator{};

        generator.setKernelName("init_array");
        generator.appendFile("kernel/common/init_array.cl");
        generator.build(&propertiesAmd.clDevice, &propertiesAmd.clContext);

        auto& clKernel{ generator.clKernel };
        OPENCL_ER(clKernel.setArg(0u, *dagCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(1u, castU32(dagItems)));

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
    }

    ////////////////////////////////////////////////////////////////////////////
    {
        common::KernelGeneratorOpenCL generator{};

        uint32_t const maxThreadByGroup{ 256u };
        uint32_t const batchGroupLane{ maxThreadByGroup / algo::progpow::LANES };

        generator.setKernelName("kawpow_lm1");

        generator.addDefine("GROUP_SIZE", maxThreadByGroup);
        generator.addDefine("REGS", algo::progpow::REGS);
        generator.addDefine("LANES", algo::progpow::LANES);
        generator.addDefine("MODULE_CACHE", algo::progpow::MODULE_CACHE);
        generator.addDefine("COUNT_DAG", algo::progpow::COUNT_DAG);
        generator.addDefine("DAG_SIZE", castU32(16777213ull / 2ull));
        generator.addDefine("BATCH_GROUP_LANE", batchGroupLane);
        generator.addDefine("SHARE_SEED_SIZE", batchGroupLane);
        generator.addDefine("SHARE_HASH0_SIZE", batchGroupLane);
        generator.addDefine("SHARE_FNV1A_SIZE", maxThreadByGroup);
        generator.addDefine("MODULE_CACHE_GROUP", maxThreadByGroup * 4u);
        generator.addDefine("MODULE_LOOP", algo::progpow::MODULE_CACHE / (maxThreadByGroup / 4u));

        generator.addInclude("kernel/common/rotate_byte.cl");
        generator.addInclude("kernel/crypto/fnv1.cl");
        generator.addInclude("kernel/crypto/keccak_f800.cl");
        generator.addInclude("kernel/crypto/kiss99.cl");

        generator.appendFile("kernel/common/result.cl");
        generator.appendFile("kernel/kawpow/sequence_dynamic.cl");
        generator.appendFile("kernel/kawpow/kawpow_lm1.cl");

        if (true == generator.build(&propertiesAmd.clDevice, &propertiesAmd.clContext))
        {
            auto& clKernel{ generator.clKernel };
            OPENCL_ER(clKernel.setArg(0u, 0ull));
            OPENCL_ER(clKernel.setArg(1u, *headerCache.getBuffer()));
            OPENCL_ER(clKernel.setArg(2u, *dagCache.getBuffer()));
            OPENCL_ER(clKernel.setArg(3u, *resultCache.getBuffer()));

            setGrid(maxThreadByGroup, 4096u);
            for (uint32_t i{ 0u }; i < 10u; ++i)
            {
                startChrono("kawpow_lm1");
                OPENCL_ER
                (
                    propertiesAmd.clQueue.enqueueNDRangeKernel
                    (
                        clKernel,
                        cl::NullRange,
                        cl::NDRange(maxThreadByGroup, 4096u, 1),
                        cl::NDRange(maxThreadByGroup, 1, 1)
                    )
                );
                OPENCL_ER(propertiesAmd.clQueue.finish());
                stopChrono(i);
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    dagCache.free();
    headerCache.free();
    resultCache.free();

    ////////////////////////////////////////////////////////////////////////////
    return true;
}

#endif
