#include <gtest/gtest.h>

#include <algo/kheavyhash/kheavyhash.hpp>
#include <algo/kheavyhash/tests/kat_vectors.hpp>


// Layer 2: heavy step (matrix multiply + XOR + KHeavyHash).
// rusty-kaspa matrix.rs::test_heavy_hash literal.
TEST(KHeavyHash, heavyHashMatchesReference)
{
    algo::kheavyhash::Hash256 const out{
        algo::kheavyhash::heavyHash(algo::kheavyhash::kat::HEAVY_TEST_MATRIX, algo::kheavyhash::kat::HEAVY_INPUT)
    };
    EXPECT_EQ(out, algo::kheavyhash::kat::HEAVY_EXPECTED);
}


// Layer 2: full pipeline incl. matrix generation. Oracle-minted (pycryptodome) vector.
TEST(KHeavyHash, calculatePowMatchesReference)
{
    algo::kheavyhash::Hash256 const out{ algo::kheavyhash::calculatePow(
        algo::kheavyhash::kat::FP_PRE,
        algo::kheavyhash::kat::FP_TIMESTAMP,
        algo::kheavyhash::kat::FP_NONCE) };
    EXPECT_EQ(out, algo::kheavyhash::kat::FP_FINAL);
}


// Accept test: pow <= target (little-endian 256-bit).
TEST(KHeavyHash, MeetsTargetCompare)
{
    EXPECT_TRUE(algo::kheavyhash::meetsTarget(algo::kheavyhash::kat::FP_FINAL, algo::kheavyhash::kat::FP_TARGET_PASS));
    EXPECT_FALSE(algo::kheavyhash::meetsTarget(algo::kheavyhash::kat::FP_FINAL, algo::kheavyhash::kat::FP_TARGET_FAIL));
}
