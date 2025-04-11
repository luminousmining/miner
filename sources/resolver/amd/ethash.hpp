#pragma once

#if defined(AMD_ENABLE)

#include <algo/dag_context.hpp>
#include <algo/hash.hpp>
#include <algo/ethash/ethash.hpp>
#include <algo/ethash/result.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <resolver/amd/amd.hpp>
#include <resolver/amd/ethash_kernel_parameter.hpp>


namespace resolver
{
    class ResolverAmdEthash : public resolver::ResolverAmd
    {
    public:
        ResolverAmdEthash() = default;
        ~ResolverAmdEthash();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

    protected:
        algo::ethash::ResultShare               resultShare{};
        resolver::amd::ethash::KernelParameters parameters{};
        algo::DagContext                        context{};
        common::KernelGeneratorOpenCL           kernelGenerator{};

        uint32_t lightCacheCountItemsGrowth{ algo::ethash::LIGHT_CACHE_COUNT_ITEMS_GROWTH };
        uint32_t lightCacheCountItemsInit{ algo::ethash::LIGHT_CACHE_COUNT_ITEMS_INIT };
        uint32_t dagCountItemsGrowth{ algo::ethash::DAG_COUNT_ITEMS_GROWTH };
        uint32_t dagCountItemsInit{ algo::ethash::DAG_COUNT_ITEMS_INIT };

        virtual bool updateContext(stratum::StratumJobInfo const& jobInfo);

        bool buildDAG();
        bool buildSearch();
        bool getResultCache(std::string const& _jobId,
                            uint32_t const extraNonceSize);
    };
}

#endif
