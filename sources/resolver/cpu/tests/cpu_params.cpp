#include <gtest/gtest.h>

#include <resolver/cpu/cpu_params.hpp>


using namespace resolver::cpu_detail;


TEST(CpuParams, nthSetBit)
{
    EXPECT_EQ(0u, nthSetBit(0b1011ull, 0u));
    EXPECT_EQ(1u, nthSetBit(0b1011ull, 1u));
    EXPECT_EQ(3u, nthSetBit(0b1011ull, 2u));
    EXPECT_EQ(64u, nthSetBit(0b1011ull, 3u)); // only 3 bits set
    EXPECT_EQ(64u, nthSetBit(0ull, 0u));
}


TEST(CpuParams, resolveWorkerCount)
{
    EXPECT_EQ(4u, resolveWorkerCount(std::optional<uint32_t>{ 4u }, 0ull, 8u)); // explicit wins
    EXPECT_EQ(3u, resolveWorkerCount(std::nullopt, 0b1011ull, 8u));             // popcount(mask)
    EXPECT_EQ(8u, resolveWorkerCount(std::nullopt, 0ull, 8u));                  // hw concurrency
    EXPECT_EQ(1u, resolveWorkerCount(std::nullopt, 0ull, 0u));                  // floor at 1
    EXPECT_EQ(8u, resolveWorkerCount(std::optional<uint32_t>{ 0u }, 0ull, 8u)); // 0 == unset
}


TEST(CpuParams, chunkRangeCoversExactly)
{
    for (uint64_t const total : { 0ull, 1ull, 7ull, 100ull, 262144ull })
    {
        for (uint32_t const n : { 1u, 3u, 8u })
        {
            uint64_t covered{ 0ull };
            uint64_t prevHi{ 0ull };
            for (uint32_t i{ 0u }; i < n; ++i)
            {
                auto const [lo, hi]{ chunkRange(total, n, i) };
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


TEST(CpuParams, parseHexMask)
{
    EXPECT_EQ(0xFFull, parseHexMask("0xFF"));
    EXPECT_EQ(0xFFull, parseHexMask("FF"));
    EXPECT_EQ(0xABCDull, parseHexMask("0xabcd"));
    EXPECT_EQ(0ull, parseHexMask(""));
    EXPECT_EQ(0ull, parseHexMask("xyz"));                                 // invalid -> 0
    EXPECT_EQ(0xFFFFFFFFFFFFFFFFull, parseHexMask("0xFFFFFFFFFFFFFFFF")); // 16 hex digits: fits
    EXPECT_EQ(0ull, parseHexMask("0x1FFFFFFFFFFFFFFFF"));                 // 17 sig digits: overflow -> 0
}
