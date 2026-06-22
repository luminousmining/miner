#include <gtest/gtest.h>

#include <algo/kheavyhash/hashers.hpp>
#include <algo/kheavyhash/tests/kat_vectors.hpp>


namespace
{
    template<size_t N>
    algo::kheavyhash::Hash256 toHash(uint8_t const (&src)[N])
    {
        static_assert(N == 32);
        algo::kheavyhash::Hash256 h{};
        for (size_t i{ 0 }; i < 32; ++i)
        {
            h[i] = src[i];
        }
        return h;
    }
}


// Layer 2: PowHash (cSHAKE256 "ProofOfWorkHash"). Matches rusty-kaspa test_pow_hash inputs.
TEST(KHeavyHashHashers, PowHashMatchesKat)
{
    algo::kheavyhash::Hash256 const pre{ toHash(algo::kheavyhash::kat::POW_KAT_PRE) };
    algo::kheavyhash::Hash256 const out{
        algo::kheavyhash::powHash(pre, algo::kheavyhash::kat::POW_KAT_TIMESTAMP, algo::kheavyhash::kat::POW_KAT_NONCE)
    };
    EXPECT_EQ(out, algo::kheavyhash::kat::POW_KAT_EXPECTED);
}


// Layer 2: KHeavyHash (cSHAKE256 "HeavyHash") over 32 bytes.
TEST(KHeavyHashHashers, KHeavyHashMatchesKat)
{
    algo::kheavyhash::Hash256 const out{ algo::kheavyhash::kHeavyHash(algo::kheavyhash::kat::HEAVY_INPUT) };
    EXPECT_EQ(out, algo::kheavyhash::kat::KHEAVY_EXPECTED);
}
