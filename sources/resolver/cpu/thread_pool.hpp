#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <boost/thread.hpp>


namespace resolver
{
    // A fixed-size pool of persistent worker threads, each optionally pinned to a logical
    // core at startup. setCallback() installs the per-chunk function; run() splits [0, count)
    // into contiguous per-worker chunks and blocks until all workers finish. Mining hits are
    // astronomically rare, so callers guard any shared write with their own mutex inside the
    // chunk function. boost::thread is used (over std::jthread) to stay consistent with the
    // rest of the threading in the project (see device_manager).
    class CpuThreadPool
    {
      public:
        using ChunkFn = std::function<void(uint64_t lo, uint64_t hi, uint32_t workerIndex)>;

        CpuThreadPool(uint32_t workerCount, uint64_t affinityMask);
        ~CpuThreadPool();

        CpuThreadPool(CpuThreadPool const&) = delete;
        CpuThreadPool& operator=(CpuThreadPool const&) = delete;

        void setCallback(ChunkFn const& fn);
        void run(uint64_t count);

      private:
        void workerLoop(uint32_t index);

        uint32_t                   poolSize;
        uint64_t                   mask;
        ChunkFn                    job{};
        uint64_t                   total{ 0ull };
        uint64_t                   generation{ 0ull };
        uint32_t                   remaining{ 0u };
        bool                       stopRequested{ false };
        boost::mutex               mutex{};
        boost::condition_variable  cvWork{};
        boost::condition_variable  cvDone{};
        std::vector<boost::thread> workers{};
    };
}
