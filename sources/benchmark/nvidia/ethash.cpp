#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <algo/fast_mod.hpp>
#include <algo/hash_utils.hpp>
#include <benchmark/workflow.hpp>
#include <benchmark/cuda/kernels.hpp>
#include <common/custom.hpp>


bool benchmark::BenchmarkWorkflow::runNvidiaEthash()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    uint64_t const dagItems{ 45023203ull };
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
    t_result* result{ nullptr };
    BENCH_INIT_RESET_RESULT(result);

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const commonLoop{ 10u };

    ////////////////////////////////////////////////////////////////////////////
    if (true == init_ethash_base(&headerHash, dagItems))
    {
        // Official etherminer implement
        RUN_BENCH
        (
            "ethash: ethash_base"s,
            commonLoop,
            128u,
            8192u,
            ethash_base(
                propertiesNvidia.cuStream,
                result,
                dagHash,
                blocks,
                threads)
        )
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == init_ethash_ethminer(dagHash, &headerHash, dagItems))
    {
        // Official etherminer implement
        RUN_BENCH
        (
            "ethash: ethminer"s,
            commonLoop,
            128u,
            8192u,
            ethash_ethminer(
                propertiesNvidia.cuStream,
                result,
                blocks,
                threads)
        )
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == init_ethash_lm1(&headerHash, dagItems))
    {
        // Custom implement
        RUN_BENCH
        (
            "ethash: ethash_lm1"s,
            commonLoop,
            128u,
            8192u,
            ethash_lm1(
                propertiesNvidia.cuStream,
                result,
                dagHash,
                blocks,
                threads)
        )
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == init_ethash_lm2(&headerHash, dagItems))
    {
        // Integer Division (fast_mod)
        RUN_BENCH
        (
            "ethash: ethash_lm2"s,
            commonLoop,
            128u,
            8192u,
            ethash_lm2(
                propertiesNvidia.cuStream,
                result,
                dagHash,
                blocks,
                threads)
        )
        BENCH_INIT_RESET_RESULT(result);
    }

    ////////////////////////////////////////////////////////////////////////////
    CU_SAFE_DELETE(dagHash);
    CU_SAFE_DELETE_HOST(result);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


#endif
