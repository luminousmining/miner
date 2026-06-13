#if defined(CUDA_ENABLE)

#include <cuda.h>
#include <cuda_runtime.h>

#include <algo/noisy_gemm/noisy_gemm.hpp>
#include <benchmark/cuda/kernels.hpp>
#include <benchmark/workflow.hpp>
#include <common/cast.hpp>
#include <common/error/cuda_error.hpp>


// Matrix dimensions for the NoisyGEMM benchmark.
// Chosen to fit comfortably in 8 GB VRAM.
// m=512, n=512, k=256: A'=128 KB, B'=128 KB, C=1 MB
static constexpr uint32_t NOISY_GEMM_BENCH_M{ 512u };
static constexpr uint32_t NOISY_GEMM_BENCH_N{ 512u };
static constexpr uint32_t NOISY_GEMM_BENCH_K{ 256u };

// Dummy seed sA (all zeros) — valid for throughput benchmarking
static constexpr uint8_t NOISY_GEMM_BENCH_SA[32]{};

// threshold = 2^256 - 1 (all tiles pass PoW) — measures raw compute throughput
static constexpr uint64_t NOISY_GEMM_BENCH_THRESHOLD[4]
{
    0xFFFFFFFFFFFFFFFFull,
    0xFFFFFFFFFFFFFFFFull,
    0xFFFFFFFFFFFFFFFFull,
    0xFFFFFFFFFFFFFFFFull
};


bool benchmark::BenchmarkWorkflow::runNvidiaNoisyGemm()
{
    ////////////////////////////////////////////////////////////////////////////
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    if (false == config.nvidia.isAlgoEnabled("noisy_gemm"))
    {
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////
    logInfo() << "Running benchmark NVIDIA NoisyGEMM (Pearl naive)";

    ////////////////////////////////////////////////////////////////////////////
    common::Dashboard            dashboard{ createNewDashboard("[NVIDIA] NOISY GEMM (Pearl)") };
    benchmark::AlgoConfig const& algo{ config.nvidia.getAlgo("noisy_gemm") };

    ////////////////////////////////////////////////////////////////////////////
    // Allocate GPU buffers shared across all three kernels
    uint64_t const sizeA{ castU64(NOISY_GEMM_BENCH_M) * castU64(NOISY_GEMM_BENCH_K) * sizeof(int8_t)  };
    uint64_t const sizeB{ castU64(NOISY_GEMM_BENCH_K) * castU64(NOISY_GEMM_BENCH_N) * sizeof(int8_t)  };
    uint64_t const sizeC{ castU64(NOISY_GEMM_BENCH_M) * castU64(NOISY_GEMM_BENCH_N) * sizeof(int32_t) };

    int8_t*  dA             { nullptr };
    int8_t*  dB             { nullptr };
    int32_t* dC             { nullptr };
    uint8_t* dSA            { nullptr };
    uint64_t* dThreshold    { nullptr };
    uint32_t* dWinningCount { nullptr };
    algo::noisy_gemm::WinningTileGpu* dWinning{ nullptr };

    CUDA_ER(cudaMalloc(&dA,            sizeA));
    CUDA_ER(cudaMalloc(&dB,            sizeB));
    CUDA_ER(cudaMalloc(&dC,            sizeC));
    CUDA_ER(cudaMalloc(&dSA,           32u));
    CUDA_ER(cudaMalloc(&dThreshold,    4u * sizeof(uint64_t)));
    CUDA_ER(cudaMalloc(&dWinningCount, sizeof(uint32_t)));
    CUDA_ER(cudaMalloc(&dWinning,      16u * sizeof(algo::noisy_gemm::WinningTileGpu)));

    // Fill A' and B' with a non-zero constant to avoid trivial zero multiplication
    CUDA_ER(cudaMemset(dA, 1, sizeA));
    CUDA_ER(cudaMemset(dB, 1, sizeB));
    CUDA_ER(cudaMemset(dC, 0, sizeC));

    CUDA_ER(cudaMemcpy(dSA,        NOISY_GEMM_BENCH_SA,        32u,                   cudaMemcpyHostToDevice));
    CUDA_ER(cudaMemcpy(dThreshold, NOISY_GEMM_BENCH_THRESHOLD, 4u * sizeof(uint64_t), cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("p1"))
    {
        KernelParams const p{ algo.resolveKernel("p1") };
        CUDA_ER(cudaMemset(dWinningCount, 0, sizeof(uint32_t)));

        RUN_BENCH(
            "noisy_gemm: naive_p1 (tm=16,tn=16,r=64)"s,
            p.loop,
            p.threads,
            p.blocks,
            pearl_naive_p1(
                propertiesNvidia.cuStream,
                dA, dB, dC,
                NOISY_GEMM_BENCH_M, NOISY_GEMM_BENCH_N, NOISY_GEMM_BENCH_K,
                dSA, dThreshold,
                dWinningCount, dWinning, 16u,
                blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("p2"))
    {
        KernelParams const p{ algo.resolveKernel("p2") };
        CUDA_ER(cudaMemset(dWinningCount, 0, sizeof(uint32_t)));

        RUN_BENCH(
            "noisy_gemm: naive_p2 (tm=32,tn=32,r=64)"s,
            p.loop,
            p.threads,
            p.blocks,
            pearl_naive_p2(
                propertiesNvidia.cuStream,
                dA, dB, dC,
                NOISY_GEMM_BENCH_M, NOISY_GEMM_BENCH_N, NOISY_GEMM_BENCH_K,
                dSA, dThreshold,
                dWinningCount, dWinning, 16u,
                blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    if (true == algo.isKernelEnabled("p3"))
    {
        KernelParams const p{ algo.resolveKernel("p3") };
        CUDA_ER(cudaMemset(dWinningCount, 0, sizeof(uint32_t)));

        RUN_BENCH(
            "noisy_gemm: naive_p3 (tm=16,tn=16,r=128)"s,
            p.loop,
            p.threads,
            p.blocks,
            pearl_naive_p3(
                propertiesNvidia.cuStream,
                dA, dB, dC,
                NOISY_GEMM_BENCH_M, NOISY_GEMM_BENCH_N, NOISY_GEMM_BENCH_K,
                dSA, dThreshold,
                dWinningCount, dWinning, 16u,
                blocks, threads))
    }

    ////////////////////////////////////////////////////////////////////////////
    CUDA_ER(cudaFree(dA));
    CUDA_ER(cudaFree(dB));
    CUDA_ER(cudaFree(dC));
    CUDA_ER(cudaFree(dSA));
    CUDA_ER(cudaFree(dThreshold));
    CUDA_ER(cudaFree(dWinningCount));
    CUDA_ER(cudaFree(dWinning));

    ////////////////////////////////////////////////////////////////////////////
    dashboards.emplace_back(dashboard);

    return true;
}

#endif
