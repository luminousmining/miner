#include <gtest/gtest.h>

#include <algo/kheavyhash/hashers.hpp>
#include "kheavyhash_test_vectors.hpp"


namespace
{
    template<size_t N>
    kheavyhash::Hash256 toHash(uint8_t const (&src)[N])
    {
        static_assert(N == 32);
        kheavyhash::Hash256 h{};
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
    kheavyhash::Hash256 const pre{ toHash(kheavyhash::kat::POW_KAT_PRE) };
    kheavyhash::Hash256 const out{
        kheavyhash::powHash(pre, kheavyhash::kat::POW_KAT_TIMESTAMP, kheavyhash::kat::POW_KAT_NONCE)
    };
    EXPECT_EQ(out, kheavyhash::kat::POW_KAT_EXPECTED);
}


// Layer 2: KHeavyHash (cSHAKE256 "HeavyHash") over 32 bytes.
TEST(KHeavyHashHashers, KHeavyHashMatchesKat)
{
    kheavyhash::Hash256 const out{ kheavyhash::kHeavyHash(kheavyhash::kat::HEAVY_INPUT) };
    EXPECT_EQ(out, kheavyhash::kat::KHEAVY_EXPECTED);
}
