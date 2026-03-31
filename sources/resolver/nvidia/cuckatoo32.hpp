#pragma once

#if defined(CUDA_ENABLE)

#include <algo/cuckatoo/cuckatoo.hpp>
#include <algo/cuckatoo/result.hpp>
#include <resolver/nvidia/cuckatoo32_kernel_parameter.hpp>
#include <resolver/nvidia/nvidia.hpp>


namespace resolver
{
    ////////////////////////////////////////////////////////////////////////////
    // Cuckatoo32 NVIDIA (CUDA) resolver – lean solver
    //
    // GPU memory layout (≈ 1.5 GB required per device):
    //   devEdgeBitmap  : NUM_EDGES / 32 uint32_t = 512 MB  – edge liveness bitmap
    //   devNodeDegree  : NUM_NODES / 16 uint32_t = 1 GB    – 2-bit degree counters
    //   devEdgeList    : MAX_EDGES_COMPACT uint32_t ≈ 4 MB  – compact surviving edges
    //   devEdgeCount   : 1 uint32_t                         – atomic compact counter
    //
    // Kernel pipeline:
    //   1. cuckatoo32Trim        – seed + 2×TRIM_ROUNDS lean trimming rounds
    //   2. cuckatoo32FindCycle   – compact edges + CPU DFS for 42-cycle
    //
    // Kernel files:
    //   sources/algo/cuckatoo/cuda/cuckatoo32.cu
    //   sources/algo/cuckatoo/cuda/siphash.cuh
    ////////////////////////////////////////////////////////////////////////////
    class ResolverNvidiaCuckatoo32 : public resolver::ResolverNvidia
    {
      public:
        ResolverNvidiaCuckatoo32();
        ~ResolverNvidiaCuckatoo32();

        bool updateMemory   (stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync    (stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync   (stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum*            stratum) final;
        void submit(stratum::StratumSmartMining* stratum) final;

      protected:
        algo::cuckatoo::ResultShare                    resultShare{};
        resolver::nvidia::cuckatoo32::KernelParameters parameters{};
    };
}

#endif
