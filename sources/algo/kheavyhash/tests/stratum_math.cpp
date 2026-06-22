#include <gtest/gtest.h>

#include <algo/kheavyhash/stratum_math.hpp>
#include <algo/kheavyhash/tests/kat_vectors.hpp>


// 4 little-endian u64 words rebuild the 32-byte pre-pow byte-for-byte.
// Words below spell out bytes 0x00..0x1f, which is exactly FP_PRE.
TEST(StratumMath, PrePowFromWordsRebuildsBytes)
{
    uint64_t const words[4]{
        0x0706050403020100ull,
        0x0f0e0d0c0b0a0908ull,
        0x1716151413121110ull,
        0x1f1e1d1c1b1a1918ull,
    };

    algo::kheavyhash::Hash256 const pre{ algo::kheavyhash::prePowFromWords(words) };

    for (int i{ 0 }; i < 32; ++i)
    {
        EXPECT_EQ(pre[i], algo::kheavyhash::kat::FP_PRE[i]) << "byte " << i;
    }
}


// diff == 1 => target == maxTarget == 2^224 - 1 == 28 bytes of 0xFF (LE).
TEST(StratumMath, DifficultyOneIsMaxTarget)
{
    algo::kheavyhash::Hash256 const target{ algo::kheavyhash::difficultyToTargetLe(1.0) };

    for (int i{ 0 }; i < 28; ++i)
    {
        EXPECT_EQ(0xFF, target[i]) << "byte " << i;
    }
    for (int i{ 28 }; i < 32; ++i)
    {
        EXPECT_EQ(0x00, target[i]) << "byte " << i;
    }
}


// diff == 2 => floor((2^224 - 1) / 2) == 2^223 - 1 == 27 x 0xFF then 0x7F (LE).
TEST(StratumMath, DifficultyTwoHalvesTarget)
{
    algo::kheavyhash::Hash256 const target{ algo::kheavyhash::difficultyToTargetLe(2.0) };

    for (int i{ 0 }; i < 27; ++i)
    {
        EXPECT_EQ(0xFF, target[i]) << "byte " << i;
    }
    EXPECT_EQ(0x7F, target[27]);
    for (int i{ 28 }; i < 32; ++i)
    {
        EXPECT_EQ(0x00, target[i]) << "byte " << i;
    }
}


// Higher difficulty => smaller target (compare as little-endian 256-bit ints).
TEST(StratumMath, HigherDifficultyShrinksTarget)
{
    algo::kheavyhash::Hash256 const t1{ algo::kheavyhash::difficultyToTargetLe(1.0) };
    algo::kheavyhash::Hash256 const t16{ algo::kheavyhash::difficultyToTargetLe(16.0) };

    bool t16Smaller{ false };
    for (int i{ 31 }; i >= 0; --i)
    {
        if (t16[i] != t1[i])
        {
            t16Smaller = t16[i] < t1[i];
            break;
        }
    }
    EXPECT_TRUE(t16Smaller);
}
