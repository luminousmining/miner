#include <gtest/gtest.h>

#include <algo/random_x/argon2d.hpp>

#include <cstring>
#include <vector>


namespace
{
    // Official RandomX test keys (UTF-8)
    static constexpr uint8_t KEY_000[]{ 't', 'e', 's', 't', ' ', 'k', 'e', 'y', ' ', '0', '0', '0' };
    static constexpr uint8_t KEY_001[]{ 't', 'e', 's', 't', ' ', 'k', 'e', 'y', ' ', '0', '0', '1' };

    static constexpr uint32_t KEY_LEN  { 12u };
    static constexpr uint64_t CACHE_BYTES{ 4194304ull * 64ull }; // 256 MiB
}


struct RandomXArgon2dTest : public testing::Test
{
    std::vector<uint8_t> cache = std::vector<uint8_t>(CACHE_BYTES, 0u);

    RandomXArgon2dTest()
    {
        algo::random_x::buildCache(cache.data(), KEY_000, KEY_LEN);
    }

    ~RandomXArgon2dTest() = default;
};


TEST_F(RandomXArgon2dTest, firstBlockNonZero)
{
    bool anyNonZero{ false };
    for (uint32_t i{ 0u }; i < 64u; ++i)
    {
        if (0u != cache[i])
        {
            anyNonZero = true;
            break;
        }
    }
    EXPECT_TRUE(anyNonZero);
}


TEST_F(RandomXArgon2dTest, lastBlockNonZero)
{
    uint64_t const offset{ CACHE_BYTES - 64u };
    bool anyNonZero{ false };
    for (uint32_t i{ 0u }; i < 64u; ++i)
    {
        if (0u != cache[offset + i])
        {
            anyNonZero = true;
            break;
        }
    }
    EXPECT_TRUE(anyNonZero);
}


TEST_F(RandomXArgon2dTest, deterministic)
{
    std::vector<uint8_t> cache2(CACHE_BYTES, 0u);
    algo::random_x::buildCache(cache2.data(), KEY_000, KEY_LEN);

    for (uint32_t i{ 0u }; i < 64u; ++i)
    {
        ASSERT_EQ(cache[i], cache2[i]) << "byte " << i;
    }
}


TEST_F(RandomXArgon2dTest, keySensitive)
{
    std::vector<uint8_t> cache2(CACHE_BYTES, 0u);
    algo::random_x::buildCache(cache2.data(), KEY_001, KEY_LEN);

    bool allSame{ true };
    for (uint32_t i{ 0u }; i < 64u; ++i)
    {
        if (cache[i] != cache2[i])
        {
            allSame = false;
            break;
        }
    }
    EXPECT_FALSE(allSame);
}
