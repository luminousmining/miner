#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <benchmark/cuda/kernels.hpp>
#include <benchmark/workflow.hpp>
#include <common/custom.hpp>


bool benchmark::BenchmarkWorkflow::runNvidiaKawpow()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark NVIDIA Kawpow";

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.nvidia.isAlgoEnabled("kawpow"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard            dashboard{ createNewDashboard("[NVIDIA] KAWPOW") };
    benchmark::AlgoConfig const& algo{ config.nvidia.getAlgo("kawpow") };

    ////////////////////////////////////////////////////////////////////////////
    uint64_t const dagItems{ 16777213ull };
    auto const     header{ algo::toHash256("71c967486cb3b70d5dfcb2ebd8eeef138453637cacbf3ccb580a41a7e96986bb") };

    ////////////////////////////////////////////////////////////////////////////
    algo::hash1024* dagHash{ nullptr };
    algo::hash256*  headerHash{ nullptr };

    CU_ALLOC(&dagHash, dagItems * algo::LEN_HASH_1024);
    CU_ALLOC(&headerHash, algo::LEN_HASH_256);

    IS_NULL(dagHash);
    IS_NULL(headerHash);

    if (false == init_array(propertiesNvidia.cuStream, dagHash->word32, dagItems))
    {
        return false;
    }

    CUDA_ER(
        cudaMemcpy((void*)headerHash->bytes, (void const*)header.bytes, algo::LEN_HASH_256, cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////
    t_result* result{ nullptr };
    if (false == initCleanResult(&result))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("kawpowminer_1"))
    {
        KernelParams const p{ algo.resolveKernel("kawpowminer_1") };
        RUN_BENCH(
            "kawpow: kawpowminer_1"s,
            p.loop,
            p.threads,
            p.blocks,
            kawpow_kawpowminer_1(
                propertiesNvidia.cuStream,
                result,
                headerHash->word32,
                dagHash->word32,
                blocks,
                threads))
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("kawpowminer_2"))
    {
        KernelParams const p{ algo.resolveKernel("kawpowminer_2") };
        RUN_BENCH(
            "kawpow: kawpowminer_2"s,
            p.loop,
            p.threads,
            p.blocks,
            kawpow_kawpowminer_2(
                propertiesNvidia.cuStream,
                result,
                headerHash->word32,
                dagHash->word32,
                blocks,
                threads))
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm1"))
    {
        KernelParams const p{ algo.resolveKernel("lm1") };
        RUN_BENCH(
            "kawpow: lm1"s,
            p.loop,
            p.threads,
            p.blocks,
            kawpow_lm1(propertiesNvidia.cuStream, result, headerHash->word32, dagHash->word32, blocks, threads))
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm2"))
    {
        KernelParams const p{ algo.resolveKernel("lm2") };
        RUN_BENCH(
            "kawpow: lm2"s,
            p.loop,
            p.threads,
            p.blocks,
            kawpow_lm2(propertiesNvidia.cuStream, result, headerHash->word32, dagHash->word32, blocks, threads))
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm3"))
    {
        KernelParams const p{ algo.resolveKernel("lm3") };
        RUN_BENCH(
            "kawpow: lm3"s,
            p.loop,
            p.threads,
            p.blocks,
            kawpow_lm3(propertiesNvidia.cuStream, result, headerHash->word32, dagHash->word32, blocks, threads))
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm4"))
    {
        KernelParams const p{ algo.resolveKernel("lm4") };
        RUN_BENCH(
            "kawpow: lm4"s,
            p.loop,
            p.threads,
            p.blocks,
            kawpow_lm4(propertiesNvidia.cuStream, result, headerHash->word32, dagHash->word32, blocks, threads))
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm5"))
    {
        KernelParams const p{ algo.resolveKernel("lm5") };
        RUN_BENCH(
            "kawpow: lm5"s,
            p.loop,
            p.threads,
            p.blocks,
            kawpow_lm5(propertiesNvidia.cuStream, result, headerHash->word32, dagHash->word32, blocks, threads))
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm6"))
    {
        KernelParams const p{ algo.resolveKernel("lm6") };
        RUN_BENCH(
            "kawpow: lm6"s,
            p.loop,
            p.threads,
            p.blocks,
            kawpow_lm6(propertiesNvidia.cuStream, result, headerHash->word32, dagHash->word32, blocks, threads))
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm7"))
    {
        KernelParams const p{ algo.resolveKernel("lm7") };
        setMultiplicator(10u);
        RUN_BENCH(
            "kawpow: lm7"s,
            p.loop,
            p.threads,
            p.blocks,
            kawpow_lm7(propertiesNvidia.cuStream, result, headerHash->word32, dagHash->word32, blocks, threads))
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm8"))
    {
        KernelParams const p{ algo.resolveKernel("lm8") };
        setMultiplicator(10u);
        RUN_BENCH(
            "kawpow: lm8"s,
            p.loop,
            p.threads,
            p.blocks,
            kawpow_lm8(propertiesNvidia.cuStream, result, headerHash->word32, dagHash->word32, blocks, threads))
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm9"))
    {
        KernelParams const p{ algo.resolveKernel("lm9") };
        RUN_BENCH(
            "kawpow: lm9"s,
            p.loop,
            p.threads,
            p.blocks,
            kawpow_lm9(propertiesNvidia.cuStream, result, headerHash->word32, dagHash->word32, blocks, threads))
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm10"))
    {
        KernelParams const p{ algo.resolveKernel("lm10") };
        setDivisor(16u);
        RUN_BENCH(
            "kawpow: lm10"s,
            p.loop,
            p.threads,
            p.blocks,
            kawpow_lm10(propertiesNvidia.cuStream, result, headerHash->word32, dagHash->word32, blocks, threads))
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm11"))
    {
        KernelParams const p{ algo.resolveKernel("lm11") };
        RUN_BENCH(
            "kawpow: lm11"s,
            p.loop,
            p.threads,
            p.blocks,
            kawpow_lm11(propertiesNvidia.cuStream, result, headerHash->word32, dagHash->word32, blocks, threads))
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    CU_SAFE_DELETE(dagHash);
    CU_SAFE_DELETE(headerHash);
    CU_SAFE_DELETE_HOST(result);

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}

#endif
