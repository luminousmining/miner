#pragma once

#include <algo/blake3/result.hpp>
#include <resolver/cpu/cpu.hpp>
#include <resolver/cpu/thread_pool.hpp>


namespace resolver
{
    class ResolverCpuBlake3 : public resolver::ResolverCpu
    {
      public:
        ResolverCpuBlake3();
        ~ResolverCpuBlake3();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;

      protected:
        algo::blake3::ResultShare resultShare{};

      private:
        // Resolved CPU pool sizing, computed once from Config so the affinity mask is parsed a
        // single time and fed to both the worker-count resolution and the pool itself.
        struct PoolConfig
        {
            uint32_t workerCount{ 1u };
            uint64_t affinityMask{ 0ull };
        };

        static PoolConfig resolvePoolConfig();
        explicit ResolverCpuBlake3(PoolConfig const poolConfig);

        // The pinned worker pool lives on the only CPU resolver that parallelizes its scan;
        // the serial progpow/kawpow CPU resolvers do not pay for an unused pool.
        resolver::CpuThreadPool pool;
    };
}
