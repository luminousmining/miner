#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <algo/fast_mod.hpp>
#include <algo/hash_utils.hpp>
#include <benchmark/benchmark.hpp>
#include <benchmark/cuda/kernels.hpp>
#include <common/custom.hpp>
#include <common/error/cuda_error.hpp>


bool benchmark::Benchmark::runNvidiaEthashLightCache()
{
    ///////////////////////////////////////////////////////////////////////////
    uint32_t* lightCache{ nullptr };
    uint32_t* seedCache{ nullptr };
    uint64_t const lightCacheNumber{ 1409017ull };
    uint64_t const lightCacheSize{ 90177088ull };
    uint32_t const commonLoop{ 1u };

    ///////////////////////////////////////////////////////////////////////////
    CU_ALLOC(&seedCache, algo::LEN_HASH_512_WORD_32 * sizeof(uint32_t));
    CU_ALLOC(&lightCache, lightCacheSize);

    ///////////////////////////////////////////////////////////////////////////
    IS_NULL(seedCache);
    IS_NULL(lightCache);

    ///////////////////////////////////////////////////////////////////////////
    // RUN_BENCH
    // (
    //     "ethash_build_light_cache_lm1",
    //     commonLoop,
    //     lightCacheNumber,
    //     1u,
    //     etash_light_cache_lm1(
    //         propertiesNvidia.cuStream,
    //         lightCache,
    //         seedCache,
    //         lightCacheNumber)
    // )

    ///////////////////////////////////////////////////////////////////////////
    RUN_BENCH
    (
        "ethash_build_light_cache_lm2",
        commonLoop,
        lightCacheNumber,
        1u,
        etash_light_cache_lm2(
            propertiesNvidia.cuStream,
            lightCache,
            seedCache,
            lightCacheNumber)
    )

    ///////////////////////////////////////////////////////////////////////////
    return true;
}

#endif
