#if defined(CPU_ENABLE)

#include <algorithm>
#include <bit>

#include <algo/bitwise.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <resolver/cpu/cpu_affinity.hpp>
#include <resolver/cpu/thread_pool.hpp>


resolver::cpu::CpuThreadPool::CpuThreadPool(uint32_t const workerCount, uint64_t const affinityMask)
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


resolver::cpu::CpuThreadPool::~CpuThreadPool()
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


void resolver::cpu::CpuThreadPool::workerLoop(uint32_t const index)
{
    if (0ull != mask)
    {
        uint32_t const population{ castU32(std::popcount(mask)) };
        uint32_t const core{ algo::nthSetBit(mask, index % population) };
        resolver::cpu::pinThisThreadToCore(core);
    }

    uint64_t lastGeneration{ 0ull };
    while (false == stopRequested.load(std::memory_order_acquire))
    {
        uint64_t jobTotal{ 0ull };
        uint64_t jobGrain{ 1ull };
        {
            UNIQUE_LOCK(mutex);
            auto const hasWork{ [&]()
                                {
                                    return generation != lastGeneration || true == stopRequested;
                                } };
            cvWork.wait(lock, hasWork);
            if (true == stopRequested)
            {
                break;
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
        for (uint64_t lo{ cursor.fetch_add(jobGrain, std::memory_order_relaxed) }; lo < jobTotal;
             lo = cursor.fetch_add(jobGrain, std::memory_order_relaxed))
        {
            uint64_t const hi{ std::min<uint64_t>(lo + jobGrain, jobTotal) };
            if (nullptr != cbJob)
            {
                cbJob(lo, hi, index);
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


void resolver::cpu::CpuThreadPool::setCallback(callbackJob const& fn)
{
    UNIQUE_LOCK(mutex);
    cbJob = fn;
}


void resolver::cpu::CpuThreadPool::runAsync(uint64_t const count, uint64_t const grain)
{
    // Single-worker (or empty) batches run inline: no dispatch, no contention. There is no
    // worker thread to run "in the background", so the inline path is unavoidably synchronous;
    // a 1-core host gets nothing from double-buffering anyway.
    if (1u >= poolSize || 0ull == count)
    {
        if (0ull != count && nullptr != cbJob)
        {
            cbJob(0ull, count, 0u);
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


void resolver::cpu::CpuThreadPool::wait()
{
    // Inline path (poolSize <= 1) already finished synchronously inside runAsync().
    if (1u >= poolSize)
    {
        return;
    }

    UNIQUE_LOCK(mutex);
    auto const isRemaining{ [&]()
                            {
                                return 0u == remaining;
                            } };
    cvDone.wait(lock, isRemaining);
}


void resolver::cpu::CpuThreadPool::run(uint64_t const count, uint64_t const grain)
{
    runAsync(count, grain);
    wait();
}

#endif
