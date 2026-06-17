#pragma once

#if defined(CPU_ENABLE)


#include <mutex>

#include <algo/blake3/result.hpp>
#include <algo/hash.hpp>
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
        // One output buffer of the double-buffer pair. executeAsync() launches the pool into the
        // idle buffer while the device reads/submits the other, so the header/target/base the
        // workers scan must be copied here by value: the worker closure outlives the
        // executeAsync() call that dispatched it, so it cannot hold references to the caller's
        // jobInfo. The hit append is astronomically rare, so a per-buffer mutex guards it with
        // negligible contention. Mirrors the GPU resultCache[2].
        struct Batch
        {
            algo::hash3072       header{};
            algo::hash256        target{};
            uint64_t             base{ 0ull };
            algo::blake3::Result result{ false, 0u, { 0ull, 0ull, 0ull, 0ull } };
            std::mutex           hitMutex{};
        };

        // Resolved CPU pool sizing, computed once from Config so the affinity mask is parsed a
        // single time and fed to both the worker-count resolution and the pool itself.
        struct PoolConfig
        {
            uint32_t workerCount{ 1u };
            uint64_t affinityMask{ 0ull };
        };

        static PoolConfig resolvePoolConfig();
        explicit ResolverCpuBlake3(PoolConfig const poolConfig);

        // Load a buffer from a job (copy header/target/base by value, reset its result) before
        // dispatching it; scan one [lo, hi) nonce slice into it; drain a completed buffer into
        // resultShare. Labeling uses the live jobInfo like the GPU executeAsync(), and submit()
        // drops anything gone stale.
        void prepareBatch(Batch& batch, stratum::StratumJobInfo const& jobInfo);
        void hashChunk(uint64_t lo, uint64_t hi, Batch& batch);
        void harvest(Batch& batch, stratum::StratumJobInfo const& jobInfo);

        // The pinned worker pool lives on the only CPU resolver that parallelizes its scan;
        // the serial progpow/kawpow CPU resolvers do not pay for an unused pool.
        resolver::cpu::CpuThreadPool threadPool;

        // Double-buffer state: executeAsync() launches into batch[currentIndex] and harvests the
        // other; inFlight tracks whether a previous async batch is still pending its wait().
        Batch    batch[2]{};
        uint32_t currentIndex{ 0u };
        bool     inFlight{ false };
    };
}

#endif
