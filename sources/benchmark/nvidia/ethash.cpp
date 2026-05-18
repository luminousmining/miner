#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <algo/fast_mod.hpp>
#include <algo/hash_utils.hpp>
#include <benchmark/cuda/kernels.hpp>
#include <benchmark/workflow.hpp>
#include <common/custom.hpp>


bool benchmark::BenchmarkWorkflow::runNvidiaEthash()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.nvidia.isAlgoEnabled("ethash"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark NVIDIA Etash";

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard            dashboard{ createNewDashboard("[NVIDIA] ETHASH") };
    benchmark::AlgoConfig const& algo{ config.nvidia.getAlgo("ethash") };

    ////////////////////////////////////////////////////////////////////////////
    uint64_t const dagItems{ 45023203ull };
    auto const     headerHash{ algo::toHash256("257cf0c2c67dd2c39842da75f97dc76d41c7cbaf31f71d5d387b16cbf3da730b") };

    ////////////////////////////////////////////////////////////////////////////
    algo::hash1024* dagHash{ nullptr };
    CU_ALLOC(&dagHash, dagItems * algo::LEN_HASH_1024);
    if (false == init_array(propertiesNvidia.cuStream, dagHash->word32, dagItems))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    t_result* result{ nullptr };
    BENCH_INIT_RESET_RESULT(result);

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("base"))
    {
        if (true == init_ethash_base(&headerHash, dagItems))
        {
            KernelParams const p{ algo.resolveKernel("base") };
            setGrid(p.threads, p.blocks);
            RUN_BENCH(
                "ethash: ethash_base"s,
                p.loop,
                threads,
                blocks,
                ethash_base(propertiesNvidia.cuStream, result, dagHash, blocks, threads))
            BENCH_INIT_RESET_RESULT(result);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("ethminer"))
    {
        if (true == init_ethash_ethminer(dagHash, &headerHash, dagItems))
        {
            KernelParams const p{ algo.resolveKernel("ethminer") };
            setGrid(p.threads, p.blocks);
            RUN_BENCH(
                "ethash: ethminer"s,
                p.loop,
                threads,
                blocks,
                ethash_ethminer(propertiesNvidia.cuStream, result, blocks, threads))
            BENCH_INIT_RESET_RESULT(result);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm1"))
    {
        if (true == init_ethash_lm1(&headerHash, dagItems))
        {
            KernelParams const p{ algo.resolveKernel("lm1") };
            setGrid(p.threads, p.blocks);
            RUN_BENCH(
                "ethash: ethash_lm1"s,
                p.loop,
                threads,
                blocks,
                ethash_lm1(propertiesNvidia.cuStream, result, dagHash, blocks, threads))
            BENCH_INIT_RESET_RESULT(result);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm2"))
    {
        if (true == init_ethash_lm2(&headerHash, dagItems))
        {
            KernelParams const p{ algo.resolveKernel("lm2") };
            setGrid(p.threads, p.blocks);
            RUN_BENCH(
                "ethash: ethash_lm2"s,
                p.loop,
                threads,
                blocks,
                ethash_lm2(propertiesNvidia.cuStream, result, dagHash, blocks, threads))
            BENCH_INIT_RESET_RESULT(result);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm3"))
    {
        if (true == init_ethash_lm3(&headerHash, dagItems))
        {
            KernelParams const p{ algo.resolveKernel("lm3") };
            setGrid(p.threads, p.blocks);
            RUN_BENCH(
                "ethash: ethash_lm3"s,
                p.loop,
                threads,
                blocks,
                ethash_lm3(propertiesNvidia.cuStream, result, dagHash, blocks, threads))
            BENCH_INIT_RESET_RESULT(result);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm4"))
    {
        if (true == init_ethash_lm4(&headerHash, dagItems))
        {
            KernelParams const p{ algo.resolveKernel("lm4") };
            setGrid(p.threads, p.blocks);
            RUN_BENCH(
                "ethash: ethash_lm4"s,
                p.loop,
                threads,
                blocks,
                ethash_lm4(propertiesNvidia.cuStream, result, dagHash, blocks, threads))
            BENCH_INIT_RESET_RESULT(result);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm5"))
    {
        if (true == init_ethash_lm5(&headerHash, dagItems))
        {
            KernelParams const p{ algo.resolveKernel("lm5") };
            setGrid(p.threads, p.blocks);
            RUN_BENCH(
                "ethash: ethash_lm5"s,
                p.loop,
                threads,
                blocks,
                ethash_lm5(propertiesNvidia.cuStream, result, dagHash, blocks, threads))
            BENCH_INIT_RESET_RESULT(result);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm6"))
    {
        if (true == init_ethash_lm6(&headerHash, dagItems))
        {
            KernelParams const p{ algo.resolveKernel("lm6") };
            setGrid(p.threads, p.blocks);
            RUN_BENCH(
                "ethash: ethash_lm6"s,
                p.loop,
                threads,
                blocks,
                ethash_lm6(propertiesNvidia.cuStream, result, dagHash, blocks, threads))
            BENCH_INIT_RESET_RESULT(result);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    CU_SAFE_DELETE(dagHash);
    CU_SAFE_DELETE_HOST(result);

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


#endif
