#if defined(CPU_ENABLE)

#include <bit>

#include <resolver/cpu/cpu_affinity.hpp>
#include <resolver/cpu/cpu_params.hpp>
#include <resolver/cpu/thread_pool.hpp>


resolver::CpuThreadPool::CpuThreadPool(uint32_t const workerCount, uint64_t const affinityMask)
    : poolSize{ (0u < workerCount) ? workerCount : 1u }, mask{ affinityMask }
{
    // poolSize == 1 runs inline in parallelFor; no worker threads are spawned.
    if (1u < poolSize)
    {
        workers.reserve(poolSize);
        for (uint32_t i{ 0u }; i < poolSize; ++i)
        {
            workers.emplace_back(
                [this, i](std::stop_token st)
                {
                    workerLoop(i, st);
                });
        }
    }
}


resolver::CpuThreadPool::~CpuThreadPool()
{
    {
        std::lock_guard<std::mutex> const lock{ mutex };
        for (std::jthread& worker : workers)
        {
            worker.request_stop();
        }
    }
    cvWork.notify_all();
    // std::jthread destructors join automatically.
}


void resolver::CpuThreadPool::workerLoop(uint32_t const index, std::stop_token stopToken)
{
    if (0ull != mask)
    {
        uint32_t const population{ static_cast<uint32_t>(std::popcount(mask)) };
        resolver::pinThisThreadToCore(resolver::cpu_detail::nthSetBit(mask, index % population));
    }

    uint64_t lastGeneration{ 0ull };
    for (;;)
    {
        ChunkFn const* fn{ nullptr };
        uint64_t       jobTotal{ 0ull };
        {
            std::unique_lock<std::mutex> lock{ mutex };
            cvWork.wait(
                lock,
                [&]
                {
                    return generation != lastGeneration || stopToken.stop_requested();
                });
            if (true == stopToken.stop_requested())
            {
                return;
            }
            lastGeneration = generation;
            fn = job;
            jobTotal = total;
        }

        auto const [lo, hi]{ resolver::cpu_detail::chunkRange(jobTotal, poolSize, index) };
        if (nullptr != fn && hi > lo)
        {
            (*fn)(lo, hi, index);
        }

        {
            std::lock_guard<std::mutex> const lock{ mutex };
            if (0u == --remaining)
            {
                cvDone.notify_one();
            }
        }
    }
}


void resolver::CpuThreadPool::parallelFor(uint64_t const count, ChunkFn const& fn)
{
    // Single-worker (or empty) batches run inline: no dispatch, no contention.
    if (1u >= poolSize || 0ull == count)
    {
        if (0ull != count)
        {
            fn(0ull, count, 0u);
        }
        return;
    }

    std::unique_lock<std::mutex> lock{ mutex };
    job = &fn;
    total = count;
    remaining = poolSize;
    ++generation;
    cvWork.notify_all();
    cvDone.wait(
        lock,
        [&]
        {
            return 0u == remaining;
        });
    job = nullptr;
}

#endif
