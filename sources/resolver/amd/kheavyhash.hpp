#pragma once

#if defined(AMD_ENABLE)

#include <algo/kheavyhash/result.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <resolver/amd/amd.hpp>
#include <resolver/amd/kheavyhash_kernel_parameter.hpp>


namespace resolver
{
    class ResolverAmdKHeavyHash : public resolver::ResolverAmd
    {
      public:
        ResolverAmdKHeavyHash();
        ~ResolverAmdKHeavyHash();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

      protected:
        algo::kheavyhash::ResultShare                resultShare{};
        resolver::amd::kheavyhash::KernelParameters  parameters{};
        common::KernelGeneratorOpenCL                kernelGenerator{};

        bool buildSearch();
        bool getResultCache(std::string const& _jobId);
    };
}

#endif
