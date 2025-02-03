#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <algo/autolykos/autolykos.hpp>
#include <algo/hash_utils.hpp>
#include <benchmark/benchmark.hpp>
#include <benchmark/cuda/kernels.hpp>
#include <common/formater_hashrate.hpp>
#include <common/log/log.hpp>
#include <common/custom.hpp>


bool benchmark::Benchmark::runNvidiaEthash()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    uint64_t const dagItems{ 45023203ull };
    uint64_t const boundary{ 10695475200ull };
    auto const headerHash
    {
        algo::toHash256("257cf0c2c67dd2c39842da75f97dc76d41c7cbaf31f71d5d387b16cbf3da730b")
    };

    ////////////////////////////////////////////////////////////////////////////
    algo::hash1024* dagHash{ nullptr };
    CU_ALLOC(&dagHash, dagItems * algo::LEN_HASH_1024);
    if (false == init_array(propertiesNvidia.cuStream, dagHash->word32, dagItems))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    t_result_64* result{ nullptr };
    if (false == initCleanResult64(&result))
    {
        return false;
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == init_ethash_ethminer(dagHash, &headerHash, dagItems, boundary))
    {
        RUN_BENCH
        (
            "ethash: ethminer"s,
            10u,
            256u,
            8192u,
            ethash_ethminer(propertiesNvidia.cuStream, result, blocks, threads)
        )
    }

    ////////////////////////////////////////////////////////////////////////////
    CU_SAFE_DELETE(dagHash);
    CU_SAFE_DELETE_HOST(result);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


bool benchmark::Benchmark::runNvidiaAutolykosv2()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

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

    algo::hash256 const header
    {
        algo::toHash256("6f109ba5226d1e0814cdeec79f1231d1d48196b5979a6d816e3621a1ef47ad80")
    };

    algo::hash256 const boundary
    {
        algo::toHash2<algo::hash256, algo::hash512>(
            algo::toLittleEndian<algo::hash512>(
                algo::decimalToHash<algo::hash512>(
                    "28948022309329048855892746252171976963209391069768726095651290785380")))
    };

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

    CUDA_ER(cudaMemcpy((void*)headerHash->bytes,
                       (void const*)header.bytes,
                       algo::LEN_HASH_256,
                       cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////
    if (true == autolykos_v2_mhssamadi_init(boundary))
    {
        if (true == autolykos_v2_mhssamadi_prehash(propertiesNvidia.cuStream,
                                                   dagHash->word32,
                                                   blocks,
                                                   threads,
                                                   period,
                                                   height))
        {
            RUN_BENCH
            (
                "autolykos_v2: mhssamadi"s,
                10u,
                64u,
                algo::autolykos_v2::NONCES_PER_ITER / 64u,
                autolykos_v2_mhssamadi(
                    propertiesNvidia.cuStream,
                    result,
                    dagHash->word32,
                    BHashes->word32,
                    headerHash->word32,
                    blocks,
                    threads,
                    period,
                    height)
            );
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == autolykos_v2_init_lm1(boundary))
    {
        if (true == autolykos_v2_prehash_lm1(propertiesNvidia.cuStream,
                                             dagHash->word32,
                                             blocks,
                                             threads,
                                             period,
                                             height))
        {
            RUN_BENCH
            (
                "autolykos_v2: lm1"s,
                10u,
                64u,
                algo::autolykos_v2::NONCES_PER_ITER / 64u,
                autolykos_v2_lm1(
                    propertiesNvidia.cuStream,
                    result,
                    dagHash->word32,
                    headerHash->word32,
                    BHashes->word32,
                    blocks,
                    threads,
                    period)
            );
        }
    }
    if (true == autolykos_v2_init_lm2(boundary))
    {
        if (true == autolykos_v2_prehash_lm1(propertiesNvidia.cuStream,
                                             dagHash->word32,
                                             blocks,
                                             threads,
                                             period,
                                             height))
        {
            // using __threadfence_block() on load global memory
            RUN_BENCH
            (
                "autolykos_v2: lm2"s,
                10u,
                64u,
                algo::autolykos_v2::NONCES_PER_ITER / 64u,
                autolykos_v2_lm2(
                    propertiesNvidia.cuStream,
                    result,
                    dagHash->word32,
                    headerHash->word32,
                    BHashes->word32,
                    blocks,
                    threads,
                    period)
            );
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    CU_SAFE_DELETE(headerHash);
    CU_SAFE_DELETE(dagHash);
    CU_SAFE_DELETE(BHashes);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


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
    // share 4096 first item of dag
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

    ////////////////////////////////////////////////////////////////////////////
    CU_SAFE_DELETE(dagHash);
    CU_SAFE_DELETE(headerHash);
    CU_SAFE_DELETE_HOST(result);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}

#endif
