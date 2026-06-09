#if defined(AMD_ENABLE)

#include <array>

#include <CL/opencl.hpp>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/kheavyhash/matrix.hpp>
#include <algo/kheavyhash/result.hpp>
#include <algo/kheavyhash/types.hpp>
#include <benchmark/workflow.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <common/opencl/buffer_mapped.hpp>
#include <common/opencl/buffer_wrapper.hpp>


bool benchmark::BenchmarkWorkflow::runAmdKHeavyHash()
{
    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark AMD kHeavyHash";

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.amd.isAlgoEnabled("kheavyhash"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard            dashboard{ createNewDashboard("[AMD] KHEAVYHASH") };
    benchmark::AlgoConfig const& algo{ config.amd.getAlgo("kheavyhash") };

    ////////////////////////////////////////////////////////////////////////////
    // kHeavyHash is not memory-hard: the only per-job state is the 64x64 nibble
    // matrix (generated host-side from the pre-pow header), the 32-byte header
    // and the 32-byte little-endian target. Any header/target works for a pure
    // throughput measurement -- the kernel runs the full
    // powHash -> heavyHash(matmul) -> kHeavyHash pipeline for every nonce no
    // matter the target. The target is left all-zero so no work-item ever
    // "meets" it: the result buffer stays untouched and atomic contention does
    // not skew the timing.
    algo::hash256 const headerHash{ algo::toHash256(
        "71c967486cb3b70d5dfcb2ebd8eeef138453637cacbf3ccb580a41a7e96986bb") };
    algo::hash256 const target{}; // value-initialised: all 32 bytes zero -> no nonce ever meets it

    ////////////////////////////////////////////////////////////////////////////
    // Host-side matrix generation (xoshiro256++ + full-rank gate) -- the CPU
    // reference the kernel is gated bit-identical against.
    ::kheavyhash::Hash256 seed{};
    for (uint32_t i{ 0u }; i < 32u; ++i)
    {
        seed[i] = headerHash.ubytes[i];
    }
    ::kheavyhash::Matrix const      matrix{ ::kheavyhash::generateMatrix(seed) };
    std::array<uint16_t, 64u * 64u> flatMatrix{};
    for (uint32_t r{ 0u }; r < 64u; ++r)
    {
        for (uint32_t c{ 0u }; c < 64u; ++c)
        {
            flatMatrix[r * 64u + c] = matrix[r][c];
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    common::opencl::Buffer<uint16_t>                       matrixCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                                                        64u * 64u * sizeof(uint16_t) };
    common::opencl::BufferMapped<algo::hash256>            headerCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                                                                        | CL_MEM_ALLOC_HOST_PTR };
    common::opencl::BufferMapped<algo::hash256>            targetCache{ CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY
                                                                        | CL_MEM_ALLOC_HOST_PTR };
    common::opencl::BufferMapped<algo::kheavyhash::Result> resultCache{ CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR };

    ////////////////////////////////////////////////////////////////////////////
    if (false == matrixCache.alloc(propertiesAmd.clContext)
        || false == headerCache.alloc(&propertiesAmd.clQueue, propertiesAmd.clContext)
        || false == targetCache.alloc(&propertiesAmd.clQueue, propertiesAmd.clContext)
        || false == resultCache.alloc(&propertiesAmd.clQueue, propertiesAmd.clContext))
    {
        logErr() << "Fail to alloc kHeavyHash benchmark buffers";
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (false == matrixCache.write(flatMatrix.data(), flatMatrix.size() * sizeof(uint16_t), &propertiesAmd.clQueue))
    {
        logErr() << "Fail to copy matrix in cache";
    }
    if (false == headerCache.setBufferDevice(&propertiesAmd.clQueue, &headerHash))
    {
        logErr() << "Fail to copy header in cache";
    }
    if (false == targetCache.setBufferDevice(&propertiesAmd.clQueue, &target))
    {
        logErr() << "Fail to copy target in cache";
    }

    ////////////////////////////////////////////////////////////////////////////
    auto benchSearch = [&](std::string const& kernelName,
                           uint32_t const     loop,
                           uint32_t const     groupSize,
                           uint32_t const     workerGroupCount) -> bool
    {
        ////////////////////////////////////////////////////////////////////////
        common::KernelGeneratorOpenCL generator{};

        ////////////////////////////////////////////////////////////////////////
        generator.setKernelName(kernelName);
        generator.addDefine("MAX_RESULT", algo::kheavyhash::MAX_RESULT);

        ////////////////////////////////////////////////////////////////////////
        // Each variant is a self-contained snapshot of one optimisation step,
        // shipped under kernel/kheavyhash/<kernel>.cl (the production `search`
        // kernel lives in the algo tree, not here).
        if (false == generator.appendFile("kernel/kheavyhash/" + kernelName + ".cl"))
        {
            return false;
        }
        if (false == generator.build(&propertiesAmd.clDevice, &propertiesAmd.clContext))
        {
            return false;
        }

        ////////////////////////////////////////////////////////////////////////
        auto&          clKernel{ generator.clKernel };
        uint64_t const timestamp{ 0x1234567890ABCDEFull };
        OPENCL_ER(clKernel.setArg(0u, *matrixCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(1u, *headerCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(2u, *targetCache.getBuffer()));
        OPENCL_ER(clKernel.setArg(3u, timestamp));
        OPENCL_ER(clKernel.setArg(4u, 0ull));
        OPENCL_ER(clKernel.setArg(5u, *resultCache.getBuffer()));

        ////////////////////////////////////////////////////////////////////////
        // One work-item per nonce: global size = groupSize * workerGroupCount,
        // which is exactly the nonce count the chrono uses for the hashrate.
        setGrid(groupSize, workerGroupCount);

        ////////////////////////////////////////////////////////////////////////
        for (uint32_t i{ 0u }; i < loop; ++i)
        {
            uint64_t const startNonce{ static_cast<uint64_t>(i) * static_cast<uint64_t>(groupSize) * workerGroupCount };
            OPENCL_ER(clKernel.setArg(4u, startNonce));

            startChrono(kernelName);
            OPENCL_ER(propertiesAmd.clQueue.enqueueNDRangeKernel(
                clKernel,
                cl::NullRange,
                cl::NDRange(static_cast<size_t>(groupSize) * workerGroupCount),
                cl::NDRange(groupSize)));
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
        benchmark::KernelParams const p{ algo.resolveKernel(name) };
        if (false == benchSearch(name, p.loop, p.threads, p.blocks))
        {
            logErr() << "kHeavyHash " << name << " benchmark aborted (GPU error)";
        }
    };

    ////////////////////////////////////////////////////////////////////////////
    // The optimisation progression that produced the production `search` kernel.
    // lm0 is the straight reference; lm4 is the kernel that ships as `search`.
    // All share the same 6-arg signature, so only the name and the
    // LDS/matmul/keccak internals differ.
    runKernel("kHeavyHash_lm0");
    runKernel("kHeavyHash_lm1");
    runKernel("kHeavyHash_lm2");
    runKernel("kHeavyHash_lm3");
    runKernel("kHeavyHash_lm4");
    runKernel("kHeavyHash_lm5");

    ////////////////////////////////////////////////////////////////////////////
    matrixCache.free();
    headerCache.free();
    targetCache.free();
    resultCache.free();

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}

#endif
