#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <algo/autolykos/autolykos.hpp>
#include <algo/hash_utils.hpp>
#include <benchmark/benchmark.hpp>
#include <benchmark/cuda/kernels.hpp>
#include <common/custom.hpp>


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


#endif
