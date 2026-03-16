#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <algo/autolykos/autolykos.hpp>
#include <algo/hash_utils.hpp>
#include <benchmark/cuda/kernels.hpp>
#include <benchmark/workflow.hpp>
#include <common/custom.hpp>


bool benchmark::BenchmarkWorkflow::runNvidiaAutolykosv2()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark NVIDIA Autolykos V2";

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.nvidia.isAlgoEnabled("autolykos_v2"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard dashboard{ createNewDashboard("[NVIDIA] AUTOLYKOS V2") };
    benchmark::AlgoConfig const& algo{ config.nvidia.getAlgo("autolykos_v2") };

    ////////////////////////////////////////////////////////////////////////////
    t_result_64* result{ nullptr };
    if (false == initCleanResult64(&result))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const height{ 3130463488u };
    uint32_t const period{ 146488965u };
    uint32_t const dagItemCount{ period * algo::autolykos_v2::NUM_SIZE_8 };

    algo::hash256 const header{ algo::toHash256("6f109ba5226d1e0814cdeec79f1231d1d48196b5979a6d816e3621a1ef47ad80") };

    algo::hash256 const boundary{ algo::toHash2<algo::hash256, algo::hash512>(algo::toLittleEndian<algo::hash512>(
        algo::decimalToHash<algo::hash512>("28948022309329048855892746252171976963209391069768726095651290785380"))) };

    ////////////////////////////////////////////////////////////////////////////
    algo::hash256* headerHash{ nullptr };
    algo::hash256* dagHash{ nullptr };
    algo::hash256* BHashes{ nullptr };

    CU_ALLOC(&headerHash, algo::LEN_HASH_256);
    CU_ALLOC(&dagHash, dagItemCount * algo::LEN_HASH_256);
    CU_ALLOC(&BHashes, algo::autolykos_v2::NONCES_PER_ITER * algo::LEN_HASH_256);

    IS_NULL(headerHash);
    IS_NULL(dagHash);
    IS_NULL(BHashes);

    CUDA_ER(
        cudaMemcpy((void*)headerHash->bytes, (void const*)header.bytes, algo::LEN_HASH_256, cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////
    // Autolykos uses algorithm-fixed grid: threads=64, blocks=NONCES_PER_ITER/64
    uint32_t const fixedThreads{ 64u };
    uint32_t const fixedBlocks{ algo::autolykos_v2::NONCES_PER_ITER / fixedThreads };

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("mhssamadi"))
    {
        if (true == autolykos_v2_mhssamadi_init(boundary))
        {
            if (true == autolykos_v2_mhssamadi_prehash(
                    propertiesNvidia.cuStream, dagHash->word32, blocks, threads, period, height))
            {
                uint32_t const loop{ algo.resolveKernel("mhssamadi").loop };
                RUN_BENCH(
                    "autolykos_v2: mhssamadi"s,
                    loop,
                    fixedThreads,
                    fixedBlocks,
                    autolykos_v2_mhssamadi(
                        propertiesNvidia.cuStream,
                        result,
                        dagHash->word32,
                        BHashes->word32,
                        headerHash->word32,
                        blocks,
                        threads,
                        period,
                        height));
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm1"))
    {
        if (true == autolykos_v2_init_lm1(boundary))
        {
            if (true == autolykos_v2_prehash_lm1(
                    propertiesNvidia.cuStream, dagHash->word32, blocks, threads, period, height))
            {
                uint32_t const loop{ algo.resolveKernel("lm1").loop };
                RUN_BENCH(
                    "autolykos_v2: lm1"s,
                    loop,
                    fixedThreads,
                    fixedBlocks,
                    autolykos_v2_lm1(
                        propertiesNvidia.cuStream,
                        result,
                        dagHash->word32,
                        headerHash->word32,
                        BHashes->word32,
                        blocks,
                        threads,
                        period));
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (algo.isKernelEnabled("lm2"))
    {
        if (true == autolykos_v2_init_lm2(boundary))
        {
            if (true == autolykos_v2_prehash_lm1(
                    propertiesNvidia.cuStream, dagHash->word32, blocks, threads, period, height))
            {
                uint32_t const loop{ algo.resolveKernel("lm2").loop };
                RUN_BENCH(
                    "autolykos_v2: lm2"s,
                    loop,
                    fixedThreads,
                    fixedBlocks,
                    autolykos_v2_lm2(
                        propertiesNvidia.cuStream,
                        result,
                        dagHash->word32,
                        headerHash->word32,
                        BHashes->word32,
                        blocks,
                        threads,
                        period));
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    CU_SAFE_DELETE(headerHash);
    CU_SAFE_DELETE(dagHash);
    CU_SAFE_DELETE(BHashes);

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


#endif
