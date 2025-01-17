#pragma once

#if defined(CUDA_ENABLE)

#include <algo/dag_context.hpp>
#include <algo/hash.hpp>
#include <algo/autolykos/result.hpp>
#include <common/kernel_generator.hpp>
#include <resolver/nvidia/nvidia.hpp>
#include <resolver/nvidia/autolykos_v2_kernel_parameter.hpp>


namespace resolver
{
    class ResolverNvidiaAutolykosV2 : public resolver::ResolverNvidia
    {
    public:
        ResolverNvidiaAutolykosV2() = default;
        virtual ~ResolverNvidiaAutolykosV2();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

    protected:
        algo::autolykos_v2::ResultShare resultShare{};
        resolver::nvidia::autolykos_v2::KernelParameters parameters{};
    };
}

#endif
