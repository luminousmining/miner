#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <benchmark/cuda/kernels.hpp>
#include <benchmark/workflow.hpp>
#include <common/custom.hpp>


bool benchmark::BenchmarkWorkflow::runNvidiaRandomX()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.nvidia.isAlgoEnabled("random_x"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark NVIDIA RandomX";

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard            dashboard{ createNewDashboard("[NVIDIA] RANDOM X") };
    benchmark::AlgoConfig const& algo{ config.nvidia.getAlgo("random_x") };

    ////////////////////////////////////////////////////////////////////////////
    // Cache (256 MiB) — built once, shared read-only by all threads
    constexpr uint64_t CACHE_SIZE{ 268435456ull };
    uint8_t* cache{ nullptr };
    CU_ALLOC(&cache, CACHE_SIZE);

    if (false == random_x_build_cache(propertiesNvidia.cuStream, cache))
    {
        CU_SAFE_DELETE(cache);
        logErr() << "RandomX: failed to build cache";
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Dataset (~2.03 GiB) — built from cache, shared read-only by all threads
    constexpr uint64_t DATASET_ITEMS{ 34078720ull };
    constexpr uint64_t DATASET_SIZE { DATASET_ITEMS * 64ull };
    uint64_t* dataset{ nullptr };
    CU_ALLOC(&dataset, DATASET_SIZE);

    if (false == random_x_build_dataset(propertiesNvidia.cuStream, cache, dataset))
    {
        CU_SAFE_DELETE(dataset);
        CU_SAFE_DELETE(cache);
        logErr() << "RandomX: failed to build dataset";
        return false;
    }

    // Cache no longer needed once the dataset is built
    CU_SAFE_DELETE(cache);

    ////////////////////////////////////////////////////////////////////////////
    // Scratchpads — 2 MiB per thread
    constexpr uint64_t SCRATCHPAD_SIZE{ 2097152ull };
    uint64_t const     scratchpadsSize{
        static_cast<uint64_t>(algo.defaults.blocks)
        * static_cast<uint64_t>(algo.defaults.threads)
        * SCRATCHPAD_SIZE
    };
    uint8_t* scratchpads{ nullptr };
    CU_ALLOC(&scratchpads, scratchpadsSize);

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("lm1"))
    {
        KernelParams const p{ algo.resolveKernel("lm1") };
        setGrid(p.threads, p.blocks);
        RUN_BENCH(
            "random_x: lm1"s,
            p.loop,
            threads,
            blocks,
            random_x_lm1(propertiesNvidia.cuStream, dataset, scratchpads, blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    CU_SAFE_DELETE(scratchpads);
    CU_SAFE_DELETE(dataset);
    dashboards.emplace_back(dashboard);

    return true;
}

#endif
