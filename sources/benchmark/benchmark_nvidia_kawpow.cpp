#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <benchmark/benchmark.hpp>
#include <benchmark/cuda/kernels.hpp>
#include <common/custom.hpp>


bool benchmark::Benchmark::runNvidiaKawpow()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    t_result* result{ nullptr };
    if (false == initCleanResult(&result))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint64_t const dagItems{ 16777213ull };
    auto const header
    {
        algo::toHash256("71c967486cb3b70d5dfcb2ebd8eeef138453637cacbf3ccb580a41a7e96986bb")
    };

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

    CUDA_ER(cudaMemcpy((void*)headerHash->bytes,
                       (void const*)header.bytes,
                       algo::LEN_HASH_256,
                       cudaMemcpyHostToDevice));

    uint32_t const commonLoop{ 10u };

    ////////////////////////////////////////////////////////////////////////////
    // Kawpowminer implementation
    RUN_BENCH
    (
        "kawpow: kawpowminer_1"s,
        commonLoop,
        256u,
        1024u,
        kawpow_kawpowminer_1(
            propertiesNvidia.cuStream,
            result,
            headerHash->word32,
            dagHash->word32,
            blocks,
            threads)
    )
    BENCH_INIT_RESET_RESULT(result);

    // Kawpowminer implementation
    RUN_BENCH
    (
        "kawpow: kawpowminer_2"s,
        commonLoop,
        256u,
        1024u,
        kawpow_kawpowminer_2(
            propertiesNvidia.cuStream,
            result,
            headerHash->word32,
            dagHash->word32,
            blocks,
            threads)
    )
    BENCH_INIT_RESET_RESULT(result);

    // Do not share 4096 first item of dag
    RUN_BENCH
    (
        "kawpow: lm1"s,
        commonLoop,
        256u,
        1024u,
        kawpow_lm1(
            propertiesNvidia.cuStream,
            result,
            headerHash->word32,
            dagHash->word32,
            blocks,
            threads)
    )
    BENCH_INIT_RESET_RESULT(result);

    // share 4096 first item of dag
    RUN_BENCH
    (
        "kawpow: lm2"s,
        commonLoop,
        256u,
        1024u,
        kawpow_lm2(propertiesNvidia.cuStream,
                   result,
                   headerHash->word32,
                   dagHash->word32,
                   blocks,
                   threads)
    )
    BENCH_INIT_RESET_RESULT(result);

    // share 4096 first item of dag
    // No unroll loop
    RUN_BENCH
    (
        "kawpow: lm3"s,
        commonLoop,
        256u,
        1024u,
        kawpow_lm3(propertiesNvidia.cuStream,
                   result,
                   headerHash->word32,
                   dagHash->word32,
                   blocks,
                   threads)
    )
    BENCH_INIT_RESET_RESULT(result);

    // share 4096 first item of dag
    // 64 regs by kernel
    RUN_BENCH
    (
        "kawpow: lm4"s,
        commonLoop,
        256u,
        1024u,
        kawpow_lm4(propertiesNvidia.cuStream,
                   result,
                   headerHash->word32,
                   dagHash->word32,
                   blocks,
                   threads)
    )
    BENCH_INIT_RESET_RESULT(result);

    // share 4096 first item of dag
    // using __threadfence_block on dag load
    RUN_BENCH
    (
        "kawpow: lm5"s,
        commonLoop,
        256u,
        1024u,
        kawpow_lm5(propertiesNvidia.cuStream,
                   result,
                   headerHash->word32,
                   dagHash->word32,
                   blocks,
                   threads)
    )
    BENCH_INIT_RESET_RESULT(result);

    // share 4096 first item of dag
    // using __threadfence_block on header initialize
    // using __threadfence_block on dag load
    RUN_BENCH
    (
        "kawpow: lm6"s,
        commonLoop,
        256u,
        1024u,
        kawpow_lm6(propertiesNvidia.cuStream,
                   result,
                   headerHash->word32,
                   dagHash->word32,
                   blocks,
                   threads)
    )
    BENCH_INIT_RESET_RESULT(result);

    // share 4096 first item of dag
    // 1 thread resolve multi nonce
    setMultiplicator(10u);
    RUN_BENCH
    (
        "kawpow: lm7"s,
        commonLoop,
        256u,
        1024u,
        kawpow_lm7(propertiesNvidia.cuStream,
                   result,
                   headerHash->word32,
                   dagHash->word32,
                   blocks,
                   threads)
    )
    BENCH_INIT_RESET_RESULT(result);

    // share 4096 first item of dag
    // 1 thread resolve multi nonce
    // using __threadfence_block on dag load
    setMultiplicator(10u);
    RUN_BENCH
    (
        "kawpow: lm8"s,
        commonLoop,
        256u,
        1024u,
        kawpow_lm8(propertiesNvidia.cuStream,
                   result,
                   headerHash->word32,
                   dagHash->word32,
                   blocks,
                   threads)
    )
    BENCH_INIT_RESET_RESULT(result);

    // share 4096 first item of dag
    // header_dag using cache read-only
    RUN_BENCH
    (
        "kawpow: lm9"s,
        commonLoop,
        256u,
        1024u,
        kawpow_lm9(propertiesNvidia.cuStream,
                   result,
                   headerHash->word32,
                   dagHash->word32,
                   blocks,
                   threads)
    )
    BENCH_INIT_RESET_RESULT(result);

    // From: lm6
    // Remove loop by LANES, using only warp parrallelism
    setDivisor(16u);
    RUN_BENCH
    (
        "kawpow: lm10"s,
        commonLoop,
        256u,
        1024u,
        kawpow_lm10(propertiesNvidia.cuStream,
                   result,
                   headerHash->word32,
                   dagHash->word32,
                   blocks,
                   threads)
    )
    BENCH_INIT_RESET_RESULT(result);

    ////////////////////////////////////////////////////////////////////////////
    CU_SAFE_DELETE(dagHash);
    CU_SAFE_DELETE(headerHash);
    CU_SAFE_DELETE_HOST(result);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


#endif
