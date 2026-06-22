#include <gtest/gtest.h>

#include <algo/kheavyhash/tests/helper.hpp>
#include <algo/kheavyhash/tests/kat_vectors.hpp>


// Layer 2: KHeavyHash (cSHAKE256 "HeavyHash") over 32 bytes.
TEST(KHeavyHashKHeavyHash, KHeavyHashMatchesKat)
{
    algo::kheavyhash::Hash256 const out{ algo::kheavyhash::kHeavyHash(algo::kheavyhash::kat::HEAVY_INPUT) };
    EXPECT_EQ(out, algo::kheavyhash::kat::KHEAVY_EXPECTED);
}
