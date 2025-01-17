#pragma once

#if defined(CUDA_ENABLE)

#include <algo/dag_context.hpp>
#include <algo/hash.hpp>
#include <algo/ethash/ethash.hpp>
#include <algo/progpow/progpow.hpp>
#include <algo/progpow/result.hpp>
#include <common/kernel_generator.hpp>
#include <resolver/nvidia/nvidia.hpp>
#include <resolver/nvidia/progpow_kernel_parameter.hpp>


namespace resolver
{
    class ResolverNvidiaProgPOW : public resolver::ResolverNvidia
    {
    public:
        ResolverNvidiaProgPOW() = default;
        virtual ~ResolverNvidiaProgPOW();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

    protected:
        algo::progpow::VERSION progpowVersion{ algo::progpow::VERSION::V_0_9_2 };
        uint64_t currentPeriod{ 0xFFFFFFFFFFFFFFFFull };
        uint32_t maxEpoch{ algo::ethash::MAX_EPOCH_NUMBER };
        uint32_t regs{ algo::progpow::REGS };
        uint32_t moduleSource{ algo::progpow::MODULE_SOURCE };
        uint32_t dagItemParents{ algo::ethash::DAG_ITEM_PARENTS };
        uint32_t dagCountItemsGrowth{ algo::ethash::DAG_COUNT_ITEMS_GROWTH };
        uint32_t dagCountItemsInit{ algo::ethash::DAG_COUNT_ITEMS_INIT };
        uint32_t countCache{ algo::progpow::v_0_9_2::COUNT_CACHE };
        uint32_t countMath{ algo::progpow::v_0_9_2::COUNT_MATH };

        algo::DagContext                            context{};
        algo::progpow::ResultShare                  resultShare{};
        common::KernelGenerator                     kernelGenerator{};
        resolver::nvidia::progpow::KernelParameters parameters{};

        bool buildSearch();
        virtual bool updateContext(stratum::StratumJobInfo const& jobInfo);
    };
}

#endif
