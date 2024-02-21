#pragma once

#include <algo/dag_context.hpp>
#include <algo/hash.hpp>
#include <algo/ethash/result.hpp>
#include <common/kernel_generator.hpp>
#include <resolver/amd/amd.hpp>
#include <resolver/amd/ethash_kernel_parameter.hpp>


namespace resolver
{
    class ResolverAmdEthash : public resolver::ResolverAmd
    {
    public:
        ResolverAmdEthash() = default;
        ~ResolverAmdEthash() = default;

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool execute(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;

    protected:
        algo::ethash::ResultShare resultShare{};
        resolver::amd::ethash::KernelParameters parameters{};
        algo::DagContext context{};
        common::KernelGenerator kernelGenerator{};

        virtual void updateContext(stratum::StratumJobInfo const& jobInfo);

        bool buildDAG();
        bool buildSearch();
        bool getResultCache(std::string const& _jobId,
                            uint32_t const extraNonceSize);
    };
}
