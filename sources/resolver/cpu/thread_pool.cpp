#if defined(CPU_ENABLE)

#include <bit>

#include <common/cast.hpp>
#include <common/custom.hpp>
#include <resolver/cpu/cpu_affinity.hpp>
#include <resolver/cpu/cpu_params.hpp>
#include <resolver/cpu/thread_pool.hpp>


resolver::CpuThreadPool::CpuThreadPool(uint32_t const workerCount, uint64_t const affinityMask)
    : poolSize{ (0u < workerCount) ? workerCount : 1u }, mask{ affinityMask }
{
    // poolSize == 1 runs inline in run(); no worker threads are spawned.
    if (1u < poolSize)
    {
        workers.reserve(poolSize);
        for (uint32_t i{ 0u }; i < poolSize; ++i)
        {
            workers.emplace_back(
                [this, i]
                {
                    workerLoop(i);
                });
        }
    }
}


resolver::CpuThreadPool::~CpuThreadPool()
{
    {
        UNIQUE_LOCK(mutex);
        stopRequested = true;
    }
    cvWork.notify_all();
    // boost::thread does not auto-join on destruction: join each worker explicitly.
    for (boost::thread& worker : workers)
    {
        if (true == worker.joinable())
        {
            worker.join();
        }
    }
}


void resolver::CpuThreadPool::workerLoop(uint32_t const index)
{
    if (0ull != mask)
    {
        uint32_t const population{ castU32(std::popcount(mask)) };
        resolver::pinThisThreadToCore(resolver::cpu::nthSetBit(mask, index % population));
    }

    uint64_t lastGeneration{ 0ull };
    for (;;)
    {
        uint64_t jobTotal{ 0ull };
        {
            UNIQUE_LOCK(mutex);
            cvWork.wait(
                lock,
                [&]
                {
                    return generation != lastGeneration || true == stopRequested;
                });
            if (true == stopRequested)
            {
                return;
            }
            lastGeneration = generation;
            jobTotal = total;
        }

        auto const [lo, hi]{ resolver::cpu::chunkRange(jobTotal, poolSize, index) };
        if (nullptr != job && hi > lo)
        {
            job(lo, hi, index);
        }

        {
            UNIQUE_LOCK(mutex);
            if (0u == --remaining)
            {
                cvDone.notify_one();
            }
        }
    }
}


void resolver::CpuThreadPool::setCallback(ChunkFn const& fn)
{
    UNIQUE_LOCK(mutex);
    job = fn;
}


void resolver::CpuThreadPool::run(uint64_t const count)
{
    // Single-worker (or empty) batches run inline: no dispatch, no contention.
    if (1u >= poolSize || 0ull == count)
    {
        if (0ull != count && nullptr != job)
        {
            job(0ull, count, 0u);
        }
        return;
    }

    UNIQUE_LOCK(mutex);
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
}

#endif
