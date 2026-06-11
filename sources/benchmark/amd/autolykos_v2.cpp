#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <algo/autolykos/autolykos.hpp>
#include <algo/autolykos/result.hpp>
#include <algo/bitwise.hpp>
#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <benchmark/workflow.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <common/opencl/buffer_mapped.hpp>
#include <common/opencl/buffer_wrapper.hpp>


////////////////////////////////////////////////////////////////////////////////
// Throughput benchmark for the production Autolykos v2 AMD pipeline. Unlike the
// ethash/progpow benchmarks there is no A/B variant to compare: the goal here is
// to close the coverage gap (autolykos_v2 was the only supported AMD algorithm
// with no in-tree benchmark) by exercising the real kernels.
//
// The DAG is built once (epoch-switch cost, untimed setup) and then the two
// hot-loop kernels are timed independently:
//   * autolykos_v2_search  - blake2b prehash over the DAG into BHashes
//   * autolykos_v2_verify  - final blake2b + boundary test
// Defines, append order and launch geometry are byte-for-byte mirrors of
// ResolverAmdAutolykosV2::buildKernelSearch / buildKernelVerify / fillDAG, so
// the measured kernels are exactly what the miner runs. The displayed grid is
// the logical nonce grid (AMD_BLOCK_DIM x AMD_NONCES_PER_ITER / AMD_BLOCK_DIM)
// so the reported hashrate is nonces/s; the enqueue itself uses the production
// strided launch (maxGroupSizeSearch / maxGroupSizeVerify work-items).
////////////////////////////////////////////////////////////////////////////////


bool benchmark::BenchmarkWorkflow::runAmdAutolykos()
{
    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark AMD Autolykos V2";

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.amd.isAlgoEnabled("autolykos_v2"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard            dashboard{ createNewDashboard("[AMD] AUTOLYKOS V2") };
    benchmark::AlgoConfig const& algo{ config.amd.getAlgo("autolykos_v2") };

    ////////////////////////////////////////////////////////////////////////////
    // Representative ERG job: ~4 GiB DAG (134,217,728 items x 32 bytes) so the
    // random DAG access pattern is not hidden by cache, well within 16 GiB VRAM.
    uint32_t const blockNumber{ 1500000u };
    uint32_t const hostPeriod{ 0x8000000u };
    uint32_t const hostHeight{ algo::be::uint32(blockNumber) };
    uint64_t const hostNonce{ 0ull };

    algo::hash256 const header{ algo::toHash256("6f109ba5226d1e0814cdeec79f1231d1d48196b5979a6d816e3621a1ef47ad80") };
    algo::hash256       boundary{};
    for (uint32_t i{ 0u }; i < algo::LEN_HASH_256 / sizeof(uint32_t); ++i)
    {
        boundary.word32[i] = 0xFFFFFFFFu;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const blockDim{ algo::autolykos_v2::AMD_BLOCK_DIM };
    uint32_t const threadsPerIter{ algo::autolykos_v2::AMD_THREADS_PER_ITER };
    uint32_t const globalSearch{ ((threadsPerIter / (blockDim * 4u)) + 1u) * blockDim };
    uint32_t const globalVerify{ ((threadsPerIter / blockDim) + 1u) * blockDim };

    ////////////////////////////////////////////////////////////////////////////
    common::opencl::Buffer<algo::u_hash256>                  BHashes{ CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS };
    common::opencl::Buffer<algo::u_hash256>                  dagCache{ CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS };
    common::opencl::BufferMapped<uint32_t>                   boundaryCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                                                              | CL_MEM_ALLOC_HOST_PTR,
                                                          algo::LEN_HASH_256 };
    common::opencl::BufferMapped<uint32_t>                   headerCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                                                            | CL_MEM_ALLOC_HOST_PTR,
                                                        algo::LEN_HASH_256 };
    common::opencl::BufferMapped<algo::autolykos_v2::Result> resultCache{ CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR };

    ////////////////////////////////////////////////////////////////////////////
    BHashes.setCapacity(algo::autolykos_v2::NONCES_PER_ITER);
    dagCache.setCapacity(hostPeriod);

    ////////////////////////////////////////////////////////////////////////////
    if (false == BHashes.alloc(propertiesAmd.clContext) || false == dagCache.alloc(propertiesAmd.clContext)
        || false == boundaryCache.alloc(&propertiesAmd.clQueue, propertiesAmd.clContext)
        || false == headerCache.alloc(&propertiesAmd.clQueue, propertiesAmd.clContext)
        || false == resultCache.alloc(&propertiesAmd.clQueue, propertiesAmd.clContext))
    {
        logErr() << "Fail to allocate Autolykos buffers";
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == headerCache.setBufferDevice(&propertiesAmd.clQueue, header.word32))
    {
        logErr() << "Fail to copy header in cache";
    }
    if (false == boundaryCache.setBufferDevice(&propertiesAmd.clQueue, boundary.word32))
    {
        logErr() << "Fail to copy boundary in cache";
    }

    ////////////////////////////////////////////////////////////////////////////
    // Build + fill the DAG once (mirrors ResolverAmdAutolykosV2::buildDAG/fillDAG).
    bool dagInitialized{ false };
    {
        ///////////////////////////////////////////////////////////////////////
        common::KernelGeneratorOpenCL generator{};

        ///////////////////////////////////////////////////////////////////////
        generator.setKernelName("autolykos_v2_build_dag");

        ///////////////////////////////////////////////////////////////////////
        if (false == generator.appendFile("kernel/common/rotate_byte.cl")
            || false == generator.appendFile("kernel/crypto/blake2b_compress.cl")
            || false == generator.appendFile("kernel/autolykos/autolykos_v2_dag.cl"))
        {
            logErr() << "Fail to assemble Autolykos DAG kernel";
            return false;
        }

        ///////////////////////////////////////////////////////////////////////
        if (true == generator.build(&propertiesAmd.clDevice, &propertiesAmd.clContext))
        {
            auto& clKernel{ generator.clKernel };
            OPENCL_ER(clKernel.setArg(0u, *dagCache.getBuffer()));
            OPENCL_ER(clKernel.setArg(1u, hostHeight));
            OPENCL_ER(clKernel.setArg(2u, hostPeriod));

            uint32_t const globalDimX{ ((hostPeriod / blockDim) + 1u) * blockDim };
            OPENCL_ER(propertiesAmd.clQueue.enqueueNDRangeKernel(
                clKernel,
                cl::NullRange,
                cl::NDRange(globalDimX, 1, 1),
                cl::NDRange(blockDim, 1, 1)));
            OPENCL_ER(propertiesAmd.clQueue.finish());

            dagInitialized = true;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Search kernel (mirrors ResolverAmdAutolykosV2::buildKernelSearch).
    common::KernelGeneratorOpenCL generatorSearch{};
    if (true == dagInitialized)
    {
        generatorSearch.setKernelName("autolykos_v2_search");
        generatorSearch.addDefine("NONCES_PER_ITER", algo::autolykos_v2::AMD_NONCES_PER_ITER);
        generatorSearch.addDefine("THREADS_PER_ITER", algo::autolykos_v2::AMD_THREADS_PER_ITER);
        generatorSearch.addDefine("K_LEN", algo::autolykos_v2::K_LEN);
        generatorSearch.addDefine("NONCE_SIZE_32", algo::autolykos_v2::NONCE_SIZE_32);
        generatorSearch.addDefine("NUM_SIZE_32", algo::autolykos_v2::NUM_SIZE_32);

        if (false == generatorSearch.appendFile("kernel/autolykos/autolykos_v2_result.cl")
            || false == generatorSearch.appendFile("kernel/common/rotate_byte.cl")
            || false == generatorSearch.appendFile("kernel/crypto/blake2b.cl")
            || false == generatorSearch.appendFile("kernel/autolykos/autolykos_v2_var_global.cl")
            || false == generatorSearch.appendFile("kernel/autolykos/autolykos_v2_search.cl")
            || false == generatorSearch.build(&propertiesAmd.clDevice, &propertiesAmd.clContext))
        {
            logErr() << "Fail to build Autolykos search kernel";
            dagInitialized = false;
        }
        else
        {
            auto& clKernel{ generatorSearch.clKernel };
            OPENCL_ER(clKernel.setArg(0u, *headerCache.getBuffer()));
            OPENCL_ER(clKernel.setArg(1u, *dagCache.getBuffer()));
            OPENCL_ER(clKernel.setArg(2u, *BHashes.getBuffer()));
            OPENCL_ER(clKernel.setArg(3u, hostNonce));
            OPENCL_ER(clKernel.setArg(4u, hostPeriod));
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Verify kernel (mirrors ResolverAmdAutolykosV2::buildKernelVerify).
    common::KernelGeneratorOpenCL generatorVerify{};
    if (true == dagInitialized)
    {
        generatorVerify.setKernelName("autolykos_v2_verify");
        generatorVerify.addDefine("NONCES_PER_ITER", algo::autolykos_v2::AMD_NONCES_PER_ITER);
        generatorVerify.addDefine("THREADS_PER_ITER", algo::autolykos_v2::AMD_THREADS_PER_ITER);
        generatorVerify.addDefine("K_LEN", algo::autolykos_v2::K_LEN);
        generatorVerify.addDefine("NONCE_SIZE_32", algo::autolykos_v2::NONCE_SIZE_32);
        generatorVerify.addDefine("NUM_SIZE_32", algo::autolykos_v2::NUM_SIZE_32);
        generatorVerify.addDefine("NUM_SIZE_8", algo::autolykos_v2::NUM_SIZE_8);

        if (false == generatorVerify.appendFile("kernel/autolykos/autolykos_v2_result.cl")
            || false == generatorVerify.appendFile("kernel/common/rotate_byte.cl")
            || false == generatorVerify.appendFile("kernel/crypto/blake2b.cl")
            || false == generatorVerify.appendFile("kernel/autolykos/autolykos_v2_var_global.cl")
            || false == generatorVerify.appendFile("kernel/autolykos/autolykos_v2_verify.cl")
            || false == generatorVerify.build(&propertiesAmd.clDevice, &propertiesAmd.clContext))
        {
            logErr() << "Fail to build Autolykos verify kernel";
            dagInitialized = false;
        }
        else
        {
            auto& clKernel{ generatorVerify.clKernel };
            OPENCL_ER(clKernel.setArg(0u, *boundaryCache.getBuffer()));
            OPENCL_ER(clKernel.setArg(1u, *dagCache.getBuffer()));
            OPENCL_ER(clKernel.setArg(2u, *BHashes.getBuffer()));
            OPENCL_ER(clKernel.setArg(3u, *resultCache.getBuffer()));
            OPENCL_ER(clKernel.setArg(4u, hostNonce));
            OPENCL_ER(clKernel.setArg(5u, hostPeriod));
            OPENCL_ER(clKernel.setArg(6u, hostHeight));
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == dagInitialized)
    {
        // One nonce iteration == AMD_NONCES_PER_ITER nonces; express the grid in
        // those terms so stopChrono reports nonces/s regardless of the strided
        // production launch geometry.
        auto const runStage =
            [&](std::string const& name, common::KernelGeneratorOpenCL& generator, uint32_t const global) -> bool
        {
            if (false == algo.isKernelEnabled(name))
            {
                return true;
            }
            benchmark::KernelParams const p{ algo.resolveKernel(name) };
            setGrid(blockDim, algo::autolykos_v2::AMD_NONCES_PER_ITER / blockDim);
            for (uint32_t i{ 0u }; i < p.loop; ++i)
            {
                startChrono(name);
                OPENCL_ER(propertiesAmd.clQueue.enqueueNDRangeKernel(
                    generator.clKernel,
                    cl::NullRange,
                    cl::NDRange(global, 1, 1),
                    cl::NDRange(blockDim, 1, 1)));
                OPENCL_ER(propertiesAmd.clQueue.finish());
                stopChrono(dashboard);
            }
            return true;
        };

        if (false == runStage("autolykos_v2_search", generatorSearch, globalSearch))
        {
            logErr() << "autolykos_v2 autolykos_v2_search benchmark aborted (GPU error)";
        }
        if (false == runStage("autolykos_v2_verify", generatorVerify, globalVerify))
        {
            logErr() << "autolykos_v2 autolykos_v2_verify benchmark aborted (GPU error)";
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    BHashes.free();
    dagCache.free();
    boundaryCache.free();
    headerCache.free();
    resultCache.free();

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}

#endif
