#include <atomic>
#include <vector>

#include <gtest/gtest.h>

#include <resolver/cpu/thread_pool.hpp>


// run() must visit every index in [0, count) exactly once, across worker counts and
// batch sizes including count < workers and count not divisible by workers.
TEST(CpuThreadPool, runCoversAllIndices)
{
    for (uint32_t const workers : { 1u, 2u, 4u, 8u })
    {
        for (uint64_t const count : { 0ull, 1ull, 5ull, 1000ull })
        {
            resolver::cpu::CpuThreadPool  pool{ workers, 0ull };
            std::vector<std::atomic<int>> seen(static_cast<size_t>(count));
            for (auto& s : seen)
            {
                s.store(0);
            }

            pool.setCallback(
                [&](uint64_t const lo, uint64_t const hi, uint32_t const)
                {
                    for (uint64_t i{ lo }; i < hi; ++i)
                    {
                        seen[static_cast<size_t>(i)].fetch_add(1);
                    }
                });
            pool.run(count);

            for (uint64_t i{ 0ull }; i < count; ++i)
            {
                EXPECT_EQ(1, seen[static_cast<size_t>(i)].load())
                    << "count=" << count << " workers=" << workers << " i=" << i;
            }
        }
    }
}


// A small explicit grain turns the single cursor into work-stealing: every index in
// [0, count) must still be visited exactly once regardless of how finely it is sliced.
TEST(CpuThreadPool, smallGrainStillCoversAllIndicesExactlyOnce)
{
    for (uint32_t const workers : { 2u, 4u, 8u })
    {
        for (uint64_t const grain : { 1ull, 3ull, 7ull })
        {
            constexpr uint64_t            count{ 1000ull };
            resolver::cpu::CpuThreadPool  pool{ workers, 0ull };
            std::vector<std::atomic<int>> seen(static_cast<size_t>(count));
            for (auto& s : seen)
            {
                s.store(0);
            }

            pool.setCallback(
                [&](uint64_t const lo, uint64_t const hi, uint32_t const)
                {
                    for (uint64_t i{ lo }; i < hi; ++i)
                    {
                        seen[static_cast<size_t>(i)].fetch_add(1);
                    }
                });
            pool.run(count, grain);

            for (uint64_t i{ 0ull }; i < count; ++i)
            {
                EXPECT_EQ(1, seen[static_cast<size_t>(i)].load())
                    << "grain=" << grain << " workers=" << workers << " i=" << i;
            }
        }
    }
}


// runAsync() must return before the batch is done; wait() then blocks until it is. After
// wait() returns, the whole range must have been covered exactly once.
TEST(CpuThreadPool, runAsyncThenWaitCoversAllIndices)
{
    constexpr uint64_t            count{ 5000ull };
    resolver::cpu::CpuThreadPool  pool{ 4u, 0ull };
    std::vector<std::atomic<int>> seen(static_cast<size_t>(count));
    for (auto& s : seen)
    {
        s.store(0);
    }

    pool.setCallback(
        [&](uint64_t const lo, uint64_t const hi, uint32_t const)
        {
            for (uint64_t i{ lo }; i < hi; ++i)
            {
                seen[static_cast<size_t>(i)].fetch_add(1);
            }
        });

    pool.runAsync(count);
    pool.wait();

    for (uint64_t i{ 0ull }; i < count; ++i)
    {
        EXPECT_EQ(1, seen[static_cast<size_t>(i)].load()) << "i=" << i;
    }
}


// Constructing with a non-zero affinity mask and running a batch must not crash and must
// still compute the correct result (pinning is best-effort and platform-dependent).
TEST(CpuThreadPool, runsWithAffinityMask)
{
    resolver::cpu::CpuThreadPool pool{ 2u, 0x3ull }; // request cores 0 and 1
    std::atomic<uint64_t>   sum{ 0ull };

    pool.setCallback(
        [&](uint64_t const lo, uint64_t const hi, uint32_t const)
        {
            uint64_t local{ 0ull };
            for (uint64_t i{ lo }; i < hi; ++i)
            {
                local += i;
            }
            sum.fetch_add(local);
        });
    pool.run(1000ull);

    EXPECT_EQ(499500ull, sum.load()); // sum of 0..999
}
