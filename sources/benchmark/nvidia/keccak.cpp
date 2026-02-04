#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include #include <benchmark/workflow.hpp>
#include <benchmark/cuda/kernels.hpp>


bool benchmark::BenchmarkWorkflow::runNvidiaKeccak()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const commonLoop{ 10u };
    uint32_t const commonThreads{ 128u };
    uint32_t const commonBlocks{ 1024u };

    ////////////////////////////////////////////////////////////////////////////
    RUN_BENCH
    (
        "keccakf800: lm1"s,
        commonLoop,
        commonThreads,
        commonBlocks,
        keccak_f800_lm1(
            propertiesNvidia.cuStream,
            blocks,
            threads)
    )

    ////////////////////////////////////////////////////////////////////////////
    // Remove __constant__
    RUN_BENCH
    (
        "keccakf800: lm2"s,
        commonLoop,
        commonThreads,
        commonBlocks,
        keccak_f800_lm2(
            propertiesNvidia.cuStream,
            blocks,
            threads)
    )

    ////////////////////////////////////////////////////////////////////////////
    // Remove __constant__
    // Remove out_base to reduce registers
    RUN_BENCH
    (
        "keccakf800: lm3"s,
        commonLoop,
        commonThreads,
        commonBlocks,
        keccak_f800_lm3(
            propertiesNvidia.cuStream,
            blocks,
            threads)
    )

    ////////////////////////////////////////////////////////////////////////////
    // Remove __constant__
    // Remove out_base to reduce registers
    // Thea step: Computing all d0-d4 first avoids reusing tmp 5 times
    RUN_BENCH
    (
        "keccakf800: lm4"s,
        commonLoop,
        commonThreads,
        commonBlocks,
        keccak_f800_lm4(
            propertiesNvidia.cuStream,
            blocks,
            threads)
    )

    ////////////////////////////////////////////////////////////////////////////
    // Remove __constant__
    // Remove out_base to reduce registers
    // Chi step: Manually unrolling the loop avoids loop overhead and i * 5u computation at each iteration
    RUN_BENCH
    (
        "keccakf800: lm5"s,
        commonLoop,
        commonThreads,
        commonBlocks,
        keccak_f800_lm5(
            propertiesNvidia.cuStream,
            blocks,
            threads)
    )

    ////////////////////////////////////////////////////////////////////////////
    // Remove __constant__
    // Remove out_base to reduce registers
    // Thea step: Computing all d0-d4 first avoids reusing tmp 5 times
    // Chi step: Manually unrolling the loop avoids loop overhead and i * 5u computation at each iteration
    RUN_BENCH
    (
        "keccakf800: lm6"s,
        commonLoop,
        commonThreads,
        commonBlocks,
        keccak_f800_lm6(
            propertiesNvidia.cuStream,
            blocks,
            threads)
    )

    ////////////////////////////////////////////////////////////////////////////
    // Remove __constant__
    // Thea step: Computing all d0-d4 first avoids reusing tmp 5 times
    RUN_BENCH
    (
        "keccakf800: lm7"s,
        commonLoop,
        commonThreads,
        commonBlocks,
        keccak_f800_lm7(
            propertiesNvidia.cuStream,
            blocks,
            threads)
    )

    ////////////////////////////////////////////////////////////////////////////
    // Remove __constant__
    // Chi step: Manually unrolling the loop avoids loop overhead and i * 5u computation at each iteration
    RUN_BENCH
    (
        "keccakf800: lm8"s,
        commonLoop,
        commonThreads,
        commonBlocks,
        keccak_f800_lm8(
            propertiesNvidia.cuStream,
            blocks,
            threads)
    )

    ////////////////////////////////////////////////////////////////////////////
    // Remove __constant__
    // Thea step: Computing all d0-d4 first avoids reusing tmp 5 times
    // Chi step: Manually unrolling the loop avoids loop overhead and i * 5u computation at each iteration
    RUN_BENCH
    (
        "keccakf800: lm9"s,
        commonLoop,
        commonThreads,
        commonBlocks,
        keccak_f800_lm9(
            propertiesNvidia.cuStream,
            blocks,
            threads)
    )
    return true;
}

#endif
