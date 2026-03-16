#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <algo/fast_mod.hpp>
#include <algo/hash_utils.hpp>
#include <benchmark/cuda/kernels.hpp>
#include <benchmark/workflow.hpp>
#include <common/custom.hpp>
#include <common/error/cuda_error.hpp>


bool benchmark::BenchmarkWorkflow::runNvidiaEthashLightCache()
{
    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark NVIDIA Light Cache";

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.nvidia.isAlgoEnabled("ethash_light_cache"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard            dashboard{ createNewDashboard("[NVIDIA] Light Cache") };
    benchmark::AlgoConfig const& algo{ config.nvidia.getAlgo("ethash_light_cache") };

    ///////////////////////////////////////////////////////////////////////////
    uint32_t*      lightCache{ nullptr };
    uint32_t*      seedCache{ nullptr };
    uint64_t const lightCacheNumber{ 1409017ull };
    uint64_t const lightCacheSize{ 90177088ull };
    uint32_t const seedCacheSize{ algo::LEN_HASH_512_WORD_32 * sizeof(uint32_t) };

    ///////////////////////////////////////////////////////////////////////////
    CU_CALLOC(&seedCache, seedCacheSize);
    CU_CALLOC(&lightCache, lightCacheSize);

    ///////////////////////////////////////////////////////////////////////////
    IS_NULL(seedCache);
    IS_NULL(lightCache);

    ///////////////////////////////////////////////////////////////////////////
    // lm1 is commented out upstream — kept disabled here intentionally

    ///////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm2"))
    {
        uint32_t const loop{ algo.resolveKernel("lm2").loop };
        RUN_BENCH(
            "ethash_build_light_cache_lm2",
            loop,
            lightCacheNumber,
            1u,
            etash_light_cache_lm2(propertiesNvidia.cuStream, lightCache, seedCache, lightCacheNumber))
        CUDA_ER(cudaMemset(seedCache, 0, seedCacheSize));
        CUDA_ER(cudaMemset(lightCache, 0, lightCacheSize));
    }

    ///////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm3"))
    {
        uint32_t const loop{ algo.resolveKernel("lm3").loop };
        RUN_BENCH(
            "ethash_build_light_cache_lm3",
            loop,
            lightCacheNumber,
            1u,
            etash_light_cache_lm3(propertiesNvidia.cuStream, lightCache, seedCache, lightCacheNumber))
        CUDA_ER(cudaMemset(seedCache, 0, seedCacheSize));
        CUDA_ER(cudaMemset(lightCache, 0, lightCacheSize));
    }

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    ///////////////////////////////////////////////////////////////////////////
    return true;
}

#endif
