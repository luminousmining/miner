#pragma once

#if defined(AMD_ENABLE)

#include <algo/cuckatoo/cuckatoo.hpp>
#include <algo/cuckatoo/result.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <resolver/amd/amd.hpp>


namespace resolver
{
    ////////////////////////////////////////////////////////////////////////////
    // Cuckatoo32 AMD (OpenCL) resolver
    //
    // GPU memory layout (≈ 4.5 GB required per device):
    //
    //   edgeBitmap  : NUM_EDGES / 8 = 512 MB
    //     Bit i is set when edge i is still alive (not yet trimmed).
    //
    //   nodeCounter : NUM_NODES bytes = 4 GB
    //     Used during the trimming phase to count node degree.
    //     A node with degree < 2 cannot belong to any cycle → mark its edges dead.
    //
    //   resultBuffer : sizeof(algo::cuckatoo::Result)
    //     Written by the cycle-detection kernel when a 42-cycle is found.
    //
    // Kernel pipeline (one mining iteration):
    //   1. seedEdges    – generate all live edges with SipHash using (pre_pow || nonce)
    //   2. trimEdges    – TRIM_ROUNDS rounds of degree-based edge removal
    //   3. findCycle    – BFS/DFS on remaining edges to locate a 42-cycle
    //
    // The actual .cl kernels live in:
    //   sources/algo/cuckatoo/opencl/cuckatoo32_seed.cl
    //   sources/algo/cuckatoo/opencl/cuckatoo32_trim.cl
    //   sources/algo/cuckatoo/opencl/cuckatoo32_cycle.cl
    ////////////////////////////////////////////////////////////////////////////
    class ResolverAmdCuckatoo32 : public resolver::ResolverAmd
    {
      public:
        ResolverAmdCuckatoo32();
        ~ResolverAmdCuckatoo32();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

      protected:
        algo::cuckatoo::ResultShare resultShare{};

        common::KernelGeneratorOpenCL kernelSeed{};
        common::KernelGeneratorOpenCL kernelTrim{};
        common::KernelGeneratorOpenCL kernelCycle{};

        // GPU buffers (allocated in updateMemory)
        cl::Buffer clEdgeBitmap{};  // 512 MB  – edge liveness bitmap
        cl::Buffer clNodeCounter{}; // 4   GB  – trimming degree counters
        cl::Buffer clResult{};      // small   – found cycle output

      private:
        bool buildKernels();
        bool allocateBuffers();
        bool runTrimming(stratum::StratumJobInfo const& jobInfo);
        bool runCycleDetection(stratum::StratumJobInfo const& jobInfo);
        bool getResult();
    };
}

#endif
