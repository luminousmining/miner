#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <benchmark/cuda/kernels.hpp>
#include <benchmark/workflow.hpp>
#include <common/custom.hpp>


bool benchmark::BenchmarkWorkflow::runNvidiaFnv1()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark NVIDIA Fnv1";

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.nvidia.isAlgoEnabled("fnv1"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard dashboard{ createNewDashboard("[NVIDIA] FNV1") };
    benchmark::AlgoConfig const& algo{ config.nvidia.getAlgo("fnv1") };

    ////////////////////////////////////////////////////////////////////////////
    benchmark::KernelParams const defaultParams{ algo.defaults };
    uint32_t* result{ nullptr };
    CU_ALLOC(&result, (defaultParams.blocks * defaultParams.threads) * sizeof(uint32_t));

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm1"))
    {
        KernelParams const p{ algo.resolveKernel("lm1") };
        RUN_BENCH("fnv1: fnv1_lm1"s, p.loop, p.threads, p.blocks,
            fnv1_lm1(propertiesNvidia.cuStream, result, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm2"))
    {
        KernelParams const p{ algo.resolveKernel("lm2") };
        RUN_BENCH("fnv1: fnv1_lm2"s, p.loop, p.threads, p.blocks,
            fnv1_lm2(propertiesNvidia.cuStream, result, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    CU_SAFE_DELETE(result);

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


#endif
