#pragma once

#if defined(CUDA_ENABLE)

#include <algo/kheavyhash/result.hpp>
#include <resolver/nvidia/kheavyhash_kernel_parameter.hpp>
#include <resolver/nvidia/nvidia.hpp>


namespace resolver
{
    class ResolverNvidiaKHeavyHash : public resolver::ResolverNvidia
    {
      public:
        ResolverNvidiaKHeavyHash();
        ~ResolverNvidiaKHeavyHash();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

      protected:
        algo::kheavyhash::ResultShare               resultShare{};
        resolver::nvidia::kheavyhash::KernelParameters parameters{};
    };
}

#endif
