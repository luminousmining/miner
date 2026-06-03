#include <gtest/gtest.h>

#include <algo/random_x/argon2d.hpp>
#include <algo/random_x/superscalar.hpp>

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


// Official test vectors from tevador/RandomX:
// key = "test key 000" (12 bytes), no padding
// https://github.com/tevador/RandomX/blob/master/src/tests/tests.cpp
TEST_F(RandomXArgon2dTest, referenceVector_000)
{
    uint64_t const* const mem{ reinterpret_cast<uint64_t const*>(cache.data()) };

    // cacheMemory[0]        == 0x191e0e1d23c02186
    // cacheMemory[1568413]  == 0xf1b62fe6210bf8b1
    // cacheMemory[33554431] == 0x1f47f056d05cd99b
    EXPECT_EQ(0x191e0e1d23c02186ULL, mem[0])        << "cacheMemory[0] mismatch";
    EXPECT_EQ(0xf1b62fe6210bf8b1ULL, mem[1568413])   << "cacheMemory[1568413] mismatch";
    EXPECT_EQ(0x1f47f056d05cd99bULL, mem[33554431])  << "cacheMemory[33554431] mismatch";
}


// CPU dataset item 0 reference vector.
// Computes dataset item 0 on CPU using executeSuperscalarProgram + cache XOR,
// then checks r[0] against the reference value from tevador/RandomX tests.
//
// Reference (interpreter mode, key = "test key 000"):
//   datasetItem[0] first uint64 == 0x680588a85ae222db
TEST_F(RandomXArgon2dTest, datasetItem0_referenceVector)
{
    // Dataset item register init constants (same as CUDA kernel)
    static constexpr uint64_t DS_INIT_MUL{ 6364136223846793005ULL };
    static constexpr uint64_t DS_XOR[7]
    {
        9298411001130361340ULL,
        12065312585734608966ULL,
        9306329213124626780ULL,
        5281919268842080866ULL,
        10536153434571861004ULL,
        3398623926847679864ULL,
        9549104520008361294ULL,
    };
    static constexpr uint64_t CACHE_ITEMS{ 4194304ULL };

    // Build SuperscalarHash programs from the same 12-byte key
    algo::random_x::SuperscalarProgram programs[algo::random_x::SUPERSCALAR_ITERS]{};
    algo::random_x::buildSuperscalarPrograms(KEY_000, programs);

    // Initialize r[0..7] for item 0
    uint64_t const r0init{ (0ULL + 1ULL) * DS_INIT_MUL };
    uint64_t r[8];
    r[0] = r0init;
    for (uint32_t i{ 1u }; i < 8u; ++i)
    {
        r[i] = r0init ^ DS_XOR[i - 1u];
    }

    // 8 rounds: SuperscalarHash + XOR with cache block
    uint64_t const* const cache64{ reinterpret_cast<uint64_t const*>(cache.data()) };
    for (uint32_t p{ 0u }; p < algo::random_x::SUPERSCALAR_ITERS; ++p)
    {
        algo::random_x::executeSuperscalarProgram(programs[p], r);
        uint64_t const  cacheIdx{ r[programs[p].addressReg] % CACHE_ITEMS };
        uint64_t const* cblk    { cache64 + cacheIdx * 8ULL };
        for (uint32_t i{ 0u }; i < 8u; ++i) { r[i] ^= cblk[i]; }
    }

    // Reference: datasetItem[0] first uint64_t == 0x680588a85ae222db
    EXPECT_EQ(0x680588a85ae222dbULL, r[0]) << "dataset item 0 r[0] mismatch";
}
