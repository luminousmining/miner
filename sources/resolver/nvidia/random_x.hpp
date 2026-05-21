#pragma once

#if defined(CUDA_ENABLE)

#include <algo/random_x/result.hpp>
#include <resolver/nvidia/random_x_kernel_parameter.hpp>
#include <resolver/nvidia/nvidia.hpp>


namespace resolver
{
    class ResolverNvidiaRandomX : public resolver::ResolverNvidia
    {
      public:
        ResolverNvidiaRandomX();
        virtual ~ResolverNvidiaRandomX();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

      protected:
        algo::random_x::ResultShare                  resultShare{};
        resolver::nvidia::random_x::KernelParameters parameters{};
    };
}

#endif
