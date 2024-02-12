#pragma once

#include <algo/dag_context.hpp>
#include <algo/hash.hpp>
#include <algo/ethash/ethash.hpp>
#include <algo/progpow/progpow.hpp>
#include <algo/progpow/result.hpp>
#include <common/kernel_generator.hpp>
#include <resolver/amd/amd.hpp>
#include <resolver/amd/progpow_kernel_parameter.hpp>


namespace resolver
{
    class ResolverAmdProgPOW : public resolver::ResolverAmd
    {
    public:
        ResolverAmdProgPOW() = default;
        ~ResolverAmdProgPOW() = default;

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool execute(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;

    protected:
        uint64_t    currentPeriod { 0ull };
        uint32_t    maxEpoch { algo::ethash::MAX_EPOCH_NUMBER };
        uint32_t    dagItemParents { algo::ethash::DAG_ITEM_PARENTS };
        uint32_t    dagCountItemsGrowth { algo::ethash::DAG_COUNT_ITEMS_GROWTH };
        uint32_t    dagCountItemsInit { algo::ethash::DAG_COUNT_ITEMS_INIT };
        uint32_t    countCache { algo::progpow::v_0_9_3::COUNT_CACHE };
        uint32_t    countMath { algo::progpow::v_0_9_3::COUNT_MATH };
        std::string kernelSHA256 { "progpow_seed.cl" };

        algo::progpow::ResultShare resultShare{};
        resolver::amd::progpow::KernelParameters parameters{};
        algo::DagContext context{};
        common::KernelGenerator kernelGenerator{};

        virtual void updateContext(stratum::StratumJobInfo const& jobInfo);

        bool buildDAG();
        bool buildSearch();
        bool getResultCache(std::string const& jobId);
    };
}
