#pragma once

#if defined(AMD_ENABLE)

#include <algo/dag_context.hpp>
#include <algo/hash.hpp>
#include <algo/ethash/ethash.hpp>
#include <algo/progpow/progpow.hpp>
#include <algo/progpow/result.hpp>
#include <common/kernel_generator/opencl.hpp>
#include <resolver/amd/amd.hpp>
#include <resolver/amd/progpow_kernel_parameter.hpp>


namespace resolver
{
    class ResolverAmdProgPOW : public resolver::ResolverAmd
    {
    public:
        ResolverAmdProgPOW() = default;
        ~ResolverAmdProgPOW();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

    protected:
        algo::progpow::VERSION progpowVersion{ algo::progpow::VERSION::V_0_9_2 };
        uint64_t currentPeriod{ 11111 };
        uint32_t maxEpoch{ algo::ethash::MAX_EPOCH_NUMBER };
        uint32_t regs{ algo::progpow::REGS };
        uint32_t moduleSource{ algo::progpow::MODULE_SOURCE };
        uint32_t lightCacheCountItemsGrowth{ algo::ethash::LIGHT_CACHE_COUNT_ITEMS_GROWTH };
        uint32_t lightCacheCountItemsInit{ algo::ethash::LIGHT_CACHE_COUNT_ITEMS_INIT };
        uint32_t dagItemParents{ algo::ethash::DAG_ITEM_PARENTS };
        uint32_t dagCountItemsGrowth{ algo::ethash::DAG_COUNT_ITEMS_GROWTH };
        uint32_t dagCountItemsInit{ algo::ethash::DAG_COUNT_ITEMS_INIT };
        uint32_t countCache{ algo::progpow::v_0_9_3::COUNT_CACHE };
        uint32_t countMath{ algo::progpow::v_0_9_3::COUNT_MATH };

        algo::progpow::ResultShare resultShare{};
        resolver::amd::progpow::KernelParameters parameters{};
        algo::DagContext context{};
        common::KernelGeneratorOpenCL kernelGenerator{};

        virtual bool updateContext(stratum::StratumJobInfo const& jobInfo);

        bool buildDAG();
        bool buildSearch();
        bool getResultCache(std::string const& jobId,
                            uint32_t const extraNonceSize);
    };
}

#endif
