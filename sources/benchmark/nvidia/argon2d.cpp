#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <benchmark/cuda/kernels.hpp>
#include <benchmark/workflow.hpp>
#include <common/error/cuda_error.hpp>
#include <common/cast.hpp>


bool benchmark::BenchmarkWorkflow::runNvidiaArgon2d()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.nvidia.isAlgoEnabled("argon2d"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark NVIDIA Argon2d";

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard            dashboard{ createNewDashboard("[NVIDIA] ARGON2D") };
    benchmark::AlgoConfig const& algo{ config.nvidia.getAlgo("argon2d") };

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("lm1"))
    {
        KernelParams const p{ algo.resolveKernel("lm1") };

        // Each thread owns 8 blocks of 128 uint64 words (8 KiB per thread).
        uint64_t* deviceMemory{ nullptr };
        uint64_t const memSize
        {
            castU64(p.threads)
                * castU64(p.blocks)
                * 8ull
                * 128ull
                * sizeof(uint64_t)
        };
        CUDA_ER(cudaMalloc(&deviceMemory, memSize));

        RUN_BENCH(
            "argon2d: lm1"s,
            p.loop,
            p.threads,
            p.blocks,
            argon2d_lm1(propertiesNvidia.cuStream, deviceMemory, blocks, threads))

        CUDA_ER(cudaFree(deviceMemory));
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("lm2"))
    {
        KernelParams const p{ algo.resolveKernel("lm2") };

        uint64_t* deviceMemory{ nullptr };
        uint64_t const memSize
        {
            static_cast<uint64_t>(p.threads)
            * static_cast<uint64_t>(p.blocks)
            * 8ull
            * 128ull
            * sizeof(uint64_t)
        };
        CUDA_ER(cudaMalloc(&deviceMemory, memSize));

        RUN_BENCH(
            "argon2d: lm2"s,
            p.loop,
            p.threads,
            p.blocks,
            argon2d_lm2(propertiesNvidia.cuStream, deviceMemory, blocks, threads))

        CUDA_ER(cudaFree(deviceMemory));
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("lm3"))
    {
        KernelParams const p{ algo.resolveKernel("lm3") };

        uint64_t* deviceMemory{ nullptr };
        uint64_t const memSize
        {
            castU64(p.threads)
                * castU64(p.blocks)
                * 8ull
                * 128ull
                * sizeof(uint64_t)
        };
        CUDA_ER(cudaMalloc(&deviceMemory, memSize));

        RUN_BENCH(
            "argon2d: lm3"s,
            p.loop,
            p.threads,
            p.blocks,
            argon2d_lm3(propertiesNvidia.cuStream, deviceMemory, blocks, threads))

        CUDA_ER(cudaFree(deviceMemory));
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("lm4"))
    {
        KernelParams const p{ algo.resolveKernel("lm4") };

        uint64_t* deviceMemory{ nullptr };
        uint64_t const memSize
        {
            castU64(p.threads)
                * castU64(p.blocks)
                * 8ull
                * 128ull
                * sizeof(uint64_t)
        };
        CUDA_ER(cudaMalloc(&deviceMemory, memSize));

        RUN_BENCH(
            "argon2d: lm4"s,
            p.loop,
            p.threads,
            p.blocks,
            argon2d_lm4(propertiesNvidia.cuStream, deviceMemory, blocks, threads))

        CUDA_ER(cudaFree(deviceMemory));
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("lm5"))
    {
        KernelParams const p{ algo.resolveKernel("lm5") };

        uint64_t* deviceMemory{ nullptr };
        uint64_t const memSize
        {
            castU64(p.threads)
                * castU64(p.blocks)
                * 8ull
                * 128ull
                * sizeof(uint64_t)
        };
        CUDA_ER(cudaMalloc(&deviceMemory, memSize));

        RUN_BENCH(
            "argon2d: lm5"s,
            p.loop,
            p.threads,
            p.blocks,
            argon2d_lm5(propertiesNvidia.cuStream, deviceMemory, blocks, threads))

        CUDA_ER(cudaFree(deviceMemory));
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("lm6"))
    {
        KernelParams const p{ algo.resolveKernel("lm6") };

        uint64_t* deviceMemory{ nullptr };
        uint64_t const memSize
        {
           castU64(p.threads)
                * castU64(p.blocks)
                * 8ull
                * 128ull
                * sizeof(uint64_t)
        };
        CUDA_ER(cudaMalloc(&deviceMemory, memSize));

        RUN_BENCH(
            "argon2d: lm6"s,
            p.loop,
            p.threads,
            p.blocks,
            argon2d_lm6(propertiesNvidia.cuStream, deviceMemory, blocks, threads))

        CUDA_ER(cudaFree(deviceMemory));
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("lm7"))
    {
        KernelParams const p{ algo.resolveKernel("lm7") };

        RUN_BENCH(
            "argon2d: lm7"s,
            p.loop,
            p.threads,
            p.blocks,
            argon2d_lm7(propertiesNvidia.cuStream, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    return true;
}

#endif
