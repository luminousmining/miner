#if defined(CPU_ENABLE)

#include <algorithm>
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
        uint64_t jobGrain{ 1ull };
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
            jobGrain = sliceGrain;
        }

        // Pull grain-sized slices from the shared cursor until [0, jobTotal) is drained. The
        // fetch_add is an atomic read-modify-write, so it always sees the latest value in the
        // modification order and no slice is ever handed out twice -- relaxed ordering is
        // enough here because the slices are independent (each worker only touches its own
        // [lo, hi) nonces; the lone shared write is the hit append, which the callback guards
        // with its own mutex).
        for (;;)
        {
            uint64_t const lo{ cursor.fetch_add(jobGrain, std::memory_order_relaxed) };
            if (lo >= jobTotal)
            {
                break;
            }
            uint64_t const hi{ std::min<uint64_t>(lo + jobGrain, jobTotal) };
            if (nullptr != job)
            {
                job(lo, hi, index);
            }
        }

        {
            UNIQUE_LOCK(mutex);
            --remaining;
            if (0u == remaining)
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


void resolver::CpuThreadPool::runAsync(uint64_t const count, uint64_t const grain)
{
    // Single-worker (or empty) batches run inline: no dispatch, no contention. There is no
    // worker thread to run "in the background", so the inline path is unavoidably synchronous;
    // a 1-core host gets nothing from double-buffering anyway.
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
    // Default grain = ceil(count / workers): every worker claims exactly one slice, so the
    // cursor reproduces the old static equal split. A smaller grain enables work-stealing.
    sliceGrain = (0ull != grain) ? grain : ((count + poolSize - 1ull) / poolSize);
    if (0ull == sliceGrain)
    {
        sliceGrain = 1ull;
    }
    cursor.store(0ull, std::memory_order_relaxed);
    remaining = poolSize;
    ++generation;
    cvWork.notify_all();
}


void resolver::CpuThreadPool::wait()
{
    // Inline path (poolSize <= 1) already finished synchronously inside runAsync().
    if (1u >= poolSize)
    {
        return;
    }

    UNIQUE_LOCK(mutex);
    cvDone.wait(
        lock,
        [&]
        {
            return 0u == remaining;
        });
}


void resolver::CpuThreadPool::run(uint64_t const count, uint64_t const grain)
{
    runAsync(count, grain);
    wait();
}

#endif
