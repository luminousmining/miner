#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>


namespace resolver
{
    // A fixed-size pool of persistent worker threads, each optionally pinned to a logical
    // core at startup. parallelFor() splits [0, count) into contiguous per-worker chunks and
    // blocks until all workers finish. Mining hits are astronomically rare, so callers guard
    // any shared write with their own mutex inside the chunk function.
    class CpuThreadPool
    {
      public:
        using ChunkFn = std::function<void(uint64_t lo, uint64_t hi, uint32_t workerIndex)>;

        CpuThreadPool(uint32_t workerCount, uint64_t affinityMask);
        ~CpuThreadPool();

        CpuThreadPool(CpuThreadPool const&) = delete;
        CpuThreadPool& operator=(CpuThreadPool const&) = delete;

        void parallelFor(uint64_t count, ChunkFn const& fn);

      private:
        void workerLoop(uint32_t index, std::stop_token stopToken);

        uint32_t                  poolSize;
        uint64_t                  mask;
        ChunkFn const*            job{ nullptr };
        uint64_t                  total{ 0ull };
        uint64_t                  generation{ 0ull };
        uint32_t                  remaining{ 0u };
        std::mutex                mutex{};
        std::condition_variable   cvWork{};
        std::condition_variable   cvDone{};
        std::vector<std::jthread> workers{};
    };
}
