#include <array>

#include <gtest/gtest.h>

#include <resolver/cpu/cpu_params.hpp>


TEST(CpuParams, resolveWorkerCount)
{
    using namespace resolver::cpu;

    EXPECT_EQ(4u, resolveWorkerCount(std::optional<uint32_t>{ 4u }, 0ull, 8u)); // explicit wins
    EXPECT_EQ(3u, resolveWorkerCount(std::nullopt, 0b1011ull, 8u));             // popcount(mask)
    EXPECT_EQ(8u, resolveWorkerCount(std::nullopt, 0ull, 8u));                  // hw concurrency
    EXPECT_EQ(1u, resolveWorkerCount(std::nullopt, 0ull, 0u));                  // floor at 1
    EXPECT_EQ(8u, resolveWorkerCount(std::optional<uint32_t>{ 0u }, 0ull, 8u)); // 0 == unset
}


TEST(CpuParams, chunkRangeCoversExactly)
{
    using namespace resolver::cpu;

    constexpr std::array<uint64_t, 5> totals{ 0ull, 1ull, 7ull, 100ull, 262144ull };
    constexpr std::array<uint32_t, 3> workerCounts{ 1u, 3u, 8u };

    for (uint64_t const total : totals)
    {
        for (uint32_t const workerCount : workerCounts)
        {
            uint64_t covered{ 0ull };
            uint64_t prevHi{ 0ull };
            for (uint32_t i{ 0u }; i < workerCount; ++i)
            {
                auto const [lo, hi]{ chunkRange(total, workerCount, i) };
                EXPECT_LE(lo, hi);
                EXPECT_EQ(prevHi, lo); // contiguous: no gap, no overlap
                prevHi = hi;
                covered += (hi - lo);
            }
            EXPECT_EQ(total, prevHi);  // last hi == total
            EXPECT_EQ(total, covered); // exact coverage
        }
    }
}
