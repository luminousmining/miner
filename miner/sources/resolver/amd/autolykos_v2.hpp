#pragma once

#include <algo/autolykos/autolykos.hpp>
#include <algo/dag_context.hpp>
#include <algo/hash.hpp>
#include <algo/autolykos/result.hpp>
#include <common/kernel_generator.hpp>
#include <resolver/amd/amd.hpp>
#include <resolver/amd/autolykos_v2_kernel_parameter.hpp>

namespace resolver
{
    class ResolverAmdAutolykosV2 : public resolver::ResolverAmd
    {
    public:
        ResolverAmdAutolykosV2() = default;
        ~ResolverAmdAutolykosV2();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool execute(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

    protected:
        algo::autolykos_v2::ResultShare resultShare{};
        resolver::amd::autolykos_v2::KernelParameters parameters{};
        common::KernelGenerator kernelGeneratorDAG{};
        common::KernelGenerator kernelGeneratorSearch{};
        common::KernelGenerator kernelGeneratorVerify{};

        bool buildDAG();
        bool buildSearch();
        bool fillDAG();
        bool getResultCache(std::string const& _jobId,
                            uint32_t const extraNonceSize,
                            uint32_t const extraNonce2Size);

    private:
        uint32_t period { 0u };
        uint32_t const maxGroupSizeSearch
        {
            ((algo::autolykos_v2::AMD_THREADS_PER_ITER / (algo::autolykos_v2::AMD_BLOCK_DIM * 4u)) + 1u)
            * algo::autolykos_v2::AMD_BLOCK_DIM
        };
        uint32_t const maxGroupSizeVerify
        {
            ((algo::autolykos_v2::AMD_THREADS_PER_ITER / algo::autolykos_v2::AMD_BLOCK_DIM) + 1u)
            * algo::autolykos_v2::AMD_BLOCK_DIM
        };

        bool buildKernelSearch();
        bool buildKernelVerify();
    };
}