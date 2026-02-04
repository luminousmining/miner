#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include #include <benchmark/workflow.hpp>
#include <benchmark/cuda/kernels.hpp>
#include <common/custom.hpp>


bool benchmark::BenchmarkWorkflow::runNvidiaFnv1()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const commonLoop{ 10u };
    uint32_t const commonThread{ 1024u };
    uint32_t const commonBlock{ 8192u };

    ////////////////////////////////////////////////////////////////////////////
    uint32_t* result{ nullptr };
    CU_ALLOC(&result, (commonBlock * commonThread) * sizeof(uint32_t));

    ////////////////////////////////////////////////////////////////////////////
    RUN_BENCH
    (
        "fnv1: fnv1_lm1"s,
        commonLoop,
        commonThread,
        commonBlock,
        fnv1_lm1(
            propertiesNvidia.cuStream,
            result,
            blocks,
            threads)
    )

    // use __umulhi instead of mult
    RUN_BENCH
    (
        "fnv1: fnv1_lm2"s,
        commonLoop,
        commonThread,
        commonBlock,
        fnv1_lm2(
            propertiesNvidia.cuStream,
            result,
            blocks,
            threads)
    )

    ////////////////////////////////////////////////////////////////////////////
    CU_SAFE_DELETE(result);

    ////////////////////////////////////////////////////////////////////////////
    return true;
}


#endif
