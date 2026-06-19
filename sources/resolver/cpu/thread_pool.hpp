#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

#include <boost/thread.hpp>


namespace resolver
{
    namespace cpu
    {
        // A fixed-size pool of persistent worker threads, each optionally pinned to a logical
        // core at startup. setCallback() installs the per-chunk function; a batch dispatches
        // [0, count) through a single atomic cursor that every worker pulls `grain`-sized slices
        // from until the range is drained. With the default grain (ceil(count / workers)) each
        // worker claims exactly one slice -- behaviour-identical to a static equal split; a smaller
        // grain turns the same loop into work-stealing (a fast core simply claims more slices),
        // which is a no-op default left for a separate benchmark to enable. Mining hits are
        // astronomically rare, so callers guard any shared write with their own mutex inside the
        // chunk function. boost::thread is used (over std::jthread) to stay consistent with the
        // rest of the threading in the project (see device_manager).
        //
        // run() dispatches and blocks until the batch finishes (used by executeSync, tests, debug).
        // runAsync() dispatches and returns immediately; the caller drains it later with wait().
        // This is the CPU mirror of the GPU two-stream double-buffer: the device reads/submits the
        // previous batch and syncs with stratum while the workers compute the next one.
        class CpuThreadPool
        {
          public:
            using callbackJob = std::function<void(uint64_t lo, uint64_t hi, uint32_t workerIndex)>;

            CpuThreadPool(uint32_t workerCount, uint64_t affinityMask);
            ~CpuThreadPool();

            CpuThreadPool(CpuThreadPool const&) = delete;
            CpuThreadPool& operator=(CpuThreadPool const&) = delete;

            void setCallback(callbackJob const& fn);
            void runAsync(uint64_t count, uint64_t grain = 0ull);
            void wait();
            void run(uint64_t count, uint64_t grain = 0ull);

          private:
            void workerLoop(uint32_t index);

            uint32_t                   poolSize{ 1u };
            uint64_t                   mask{ 0ull };
            callbackJob                cbJob{};
            uint64_t                   total{ 0ull };
            uint64_t                   sliceGrain{ 1ull };
            std::atomic<uint64_t>      cursor{ 0ull };
            uint64_t                   generation{ 0ull };
            uint32_t                   remaining{ 0u };
            std::atomic<bool>          stopRequested{ false };
            boost::mutex               mutex{};
            boost::condition_variable  cvWork{};
            boost::condition_variable  cvDone{};
            std::vector<boost::thread> workers{};
        };
    }
}
