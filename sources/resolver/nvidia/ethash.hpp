#pragma once

#if defined(CUDA_ENABLE)

#include <algo/dag_context.hpp>
#include <algo/hash.hpp>
#include <algo/ethash/result.hpp>
#include <resolver/nvidia/nvidia.hpp>
#include <resolver/nvidia/ethash_kernel_parameter.hpp>


namespace resolver
{
    class ResolverNvidiaEthash : public resolver::ResolverNvidia
    {
    public:
        ResolverNvidiaEthash() = default;
        ~ResolverNvidiaEthash();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

    protected:
        algo::ethash::ResultShare resultShare{};
        resolver::nvidia::ethash::KernelParameters parameters{};
        algo::DagContext context{};

        virtual bool updateContext(stratum::StratumJobInfo const& jobInfo);
    };
}

#endif
