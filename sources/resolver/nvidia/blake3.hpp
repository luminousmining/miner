#pragma once

#if defined(CUDA_ENABLE)

#include <algo/dag_context.hpp>
#include <algo/hash.hpp>
#include <algo/ethash/result.hpp>
#include <resolver/nvidia/nvidia.hpp>
#include <resolver/nvidia/blake3_kernel_parameter.hpp>


namespace resolver
{
    class ResolverNvidiaBlake3 : public resolver::ResolverNvidia
    {
    public:
        ResolverNvidiaBlake3();
        ~ResolverNvidiaBlake3();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

    protected:
        algo::blake3::ResultShare resultShare{};
        resolver::nvidia::blake3::KernelParameters parameters{};
    };
}

#endif
