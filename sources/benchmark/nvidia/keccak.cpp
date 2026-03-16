#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <benchmark/cuda/kernels.hpp>
#include <benchmark/workflow.hpp>


bool benchmark::BenchmarkWorkflow::runNvidiaKeccak()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark NVIDIA Keccak";

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.nvidia.isAlgoEnabled("keccak"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard            dashboard{ createNewDashboard("[NVIDIA] KECCAK") };
    benchmark::AlgoConfig const& algo{ config.nvidia.getAlgo("keccak") };

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm1"))
    {
        KernelParams const p{ algo.resolveKernel("lm1") };
        RUN_BENCH(
            "keccakf800: lm1"s,
            p.loop,
            p.threads,
            p.blocks,
            keccak_f800_lm1(propertiesNvidia.cuStream, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm2"))
    {
        KernelParams const p{ algo.resolveKernel("lm2") };
        RUN_BENCH(
            "keccakf800: lm2"s,
            p.loop,
            p.threads,
            p.blocks,
            keccak_f800_lm2(propertiesNvidia.cuStream, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm3"))
    {
        KernelParams const p{ algo.resolveKernel("lm3") };
        RUN_BENCH(
            "keccakf800: lm3"s,
            p.loop,
            p.threads,
            p.blocks,
            keccak_f800_lm3(propertiesNvidia.cuStream, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm4"))
    {
        KernelParams const p{ algo.resolveKernel("lm4") };
        RUN_BENCH(
            "keccakf800: lm4"s,
            p.loop,
            p.threads,
            p.blocks,
            keccak_f800_lm4(propertiesNvidia.cuStream, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm5"))
    {
        KernelParams const p{ algo.resolveKernel("lm5") };
        RUN_BENCH(
            "keccakf800: lm5"s,
            p.loop,
            p.threads,
            p.blocks,
            keccak_f800_lm5(propertiesNvidia.cuStream, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm6"))
    {
        KernelParams const p{ algo.resolveKernel("lm6") };
        RUN_BENCH(
            "keccakf800: lm6"s,
            p.loop,
            p.threads,
            p.blocks,
            keccak_f800_lm6(propertiesNvidia.cuStream, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm7"))
    {
        KernelParams const p{ algo.resolveKernel("lm7") };
        RUN_BENCH(
            "keccakf800: lm7"s,
            p.loop,
            p.threads,
            p.blocks,
            keccak_f800_lm7(propertiesNvidia.cuStream, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm8"))
    {
        KernelParams const p{ algo.resolveKernel("lm8") };
        RUN_BENCH(
            "keccakf800: lm8"s,
            p.loop,
            p.threads,
            p.blocks,
            keccak_f800_lm8(propertiesNvidia.cuStream, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm9"))
    {
        KernelParams const p{ algo.resolveKernel("lm9") };
        RUN_BENCH(
            "keccakf800: lm9"s,
            p.loop,
            p.threads,
            p.blocks,
            keccak_f800_lm9(propertiesNvidia.cuStream, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    return true;
}

#endif
