#pragma once

#if defined(AMD_ENABLE)

#include <algo/blake3/result.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <resolver/amd/amd.hpp>
#include <resolver/amd/blake3_kernel_parameter.hpp>


namespace resolver
{
    class ResolverAmdBlake3 : public resolver::ResolverAmd
    {
      public:
        ResolverAmdBlake3();
        ~ResolverAmdBlake3();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

      protected:
        algo::blake3::ResultShare              resultShare{};
        resolver::amd::blake3::KernelParameters parameters{};
        common::KernelGeneratorOpenCL          kernelGenerator{};

        bool buildSearch();
        bool getResultCache(std::string const& _jobId, uint32_t const fromGroup, uint32_t const toGroup,
                            uint32_t const extraNonceSize);
    };
}

#endif
