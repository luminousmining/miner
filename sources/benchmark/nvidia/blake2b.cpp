#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <benchmark/cuda/kernels.hpp>
#include <benchmark/workflow.hpp>


bool benchmark::BenchmarkWorkflow::runNvidiaBlake2b()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.nvidia.isAlgoEnabled("blake2b"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark NVIDIA Blake2b";

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard            dashboard{ createNewDashboard("[NVIDIA] BLAKE2B") };
    benchmark::AlgoConfig const& algo{ config.nvidia.getAlgo("blake2b") };

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("lm1"))
    {
        KernelParams const p{ algo.resolveKernel("lm1") };
        setGrid(p.threads, p.blocks);
        RUN_BENCH(
            "blake2b: lm1"s,
            p.loop,
            threads,
            blocks,
            blake2b_lm1(propertiesNvidia.cuStream, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("lm2"))
    {
        KernelParams const p{ algo.resolveKernel("lm2") };
        setGrid(p.threads, p.blocks);
        RUN_BENCH(
            "blake2b: lm2"s,
            p.loop,
            threads,
            blocks,
            blake2b_lm2(propertiesNvidia.cuStream, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("lm3"))
    {
        KernelParams const p{ algo.resolveKernel("lm3") };
        setGrid(p.threads, p.blocks);
        RUN_BENCH(
            "blake2b: lm3"s,
            p.loop,
            threads,
            blocks,
            blake2b_lm3(propertiesNvidia.cuStream, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("lm4"))
    {
        KernelParams const p{ algo.resolveKernel("lm4") };
        setGrid(p.threads, p.blocks);
        RUN_BENCH(
            "blake2b: lm4"s,
            p.loop,
            threads,
            blocks,
            blake2b_lm4(propertiesNvidia.cuStream, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("lm5"))
    {
        KernelParams const p{ algo.resolveKernel("lm5") };
        setGrid(p.threads, p.blocks);
        RUN_BENCH(
            "blake2b: lm5"s,
            p.loop,
            threads,
            blocks,
            blake2b_lm5(propertiesNvidia.cuStream, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    return true;
}

#endif
