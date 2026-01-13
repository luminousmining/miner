#include <gtest/gtest.h>

#include "algo/fast_mod.hpp"

struct FastModTest : public ::testing::Test
{
    void compareWithBasicMod(
        uint32_t const divisor,
        uint32_t const value)
    {
        auto const fd{ initFastMod(divisor) };
        auto const fastResult{ fastMod(fd, value) };
        auto const basicResult{ value % divisor };

        ASSERT_EQ(fastResult, basicResult)
            << "Mismatch for value=" << value << " divisor=" << divisor
            << " fast=" << fastResult << " basic=" << basicResult;
    }
};



TEST_F(FastModTest, SmallDivisors)
{
    for (uint32_t d{ 2 }; d <= 100; ++d)
    {
        for (uint32_t v{ 0 }; v < 1000; ++v)
        {
            ASSERT_NO_FATAL_FAILURE(compareWithBasicMod(d, v));
        }
    }
}


TEST_F(FastModTest, PowerOfTwo)
{
    uint32_t const divisors[]{ 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };

    for (auto const d : divisors)
    {
        for (uint32_t v{ 0 }; v < 10000; ++v)
        {
            ASSERT_NO_FATAL_FAILURE(compareWithBasicMod(d, v));
        }
    }
}


TEST_F(FastModTest, LargeDivisors)
{
    uint32_t const divisors[]{ 1000000, 12345678, 99999999, 0x7FFFFFFF };

    for (auto const d : divisors)
    {
        for (uint32_t v{ 0 }; v < 100000; ++v)
        {
            ASSERT_NO_FATAL_FAILURE(compareWithBasicMod(d, v));
        }
    }
}


TEST_F(FastModTest, LargeValues)
{
    uint32_t const divisors[]{ 7, 13, 1000, 65537 };

    for (auto const d : divisors)
    {
        uint32_t const values[]{
            0xFFFFFFFF,
            0xFFFFFFFE,
            0x80000000,
            0x7FFFFFFF,
            0x12345678,
            0xDEADBEEF
        };

        for (auto const v : values)
        {
            ASSERT_NO_FATAL_FAILURE(compareWithBasicMod(d, v));
        }
    }
}


TEST_F(FastModTest, EthereumDagSizes)
{
    uint32_t const dagSizes[]{
        1073739904,
        1082130304,
        1090514816,
        1098906752
    };

    for (auto const d : dagSizes)
    {
        auto const fd{ initFastMod(d) };

        for (uint32_t i{ 0 }; i < 100000; ++i)
        {
            uint32_t const v{ i * 12345 };
            auto const fastResult{ fastMod(fd, v) };
            auto const basicResult{ v % d };

            EXPECT_EQ(fastResult, basicResult);
        }
    }
}


TEST_F(FastModTest, BoundaryValues)
{
    uint32_t const divisors[]{ 3, 7, 100, 1000 };

    for (auto const d : divisors)
    {
        compareWithBasicMod(d, 0);
        compareWithBasicMod(d, 1);
        compareWithBasicMod(d, d - 1);
        compareWithBasicMod(d, d);
        compareWithBasicMod(d, d + 1);
        compareWithBasicMod(d, d * 2);
        compareWithBasicMod(d, d * 2 - 1);
        compareWithBasicMod(d, d * 2 + 1);
    }
}


TEST_F(FastModTest, RandomValues)
{
    uint32_t seed{ 0xDEADBEEF };

    auto const lcg = [&seed]() -> uint32_t
    {
        seed = seed * 1103515245 + 12345;
        return seed;
    };

    for (uint32_t i{ 0 }; i < 1000; ++i)
    {
        auto const d{ (lcg() % 0x7FFFFFFE) + 2 };
        auto const v{ lcg() };

        ASSERT_NO_FATAL_FAILURE(compareWithBasicMod(d, v));
    }
}


TEST_F(FastModTest, InitFastModEdgeCases)
{
    {
        auto const fd{ initFastMod(0) };
        EXPECT_EQ(fd.divisor, 0u);
        EXPECT_EQ(fd.magic, 0u);
        EXPECT_EQ(fd.shift, 0u);
    }

    {
        auto const fd{ initFastMod(1) };
        EXPECT_EQ(fd.divisor, 1u);
        EXPECT_EQ(fd.magic, 0u);
        EXPECT_EQ(fd.shift, 0u);
    }
}
