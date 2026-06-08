#include <gtest/gtest.h>

#include <algo/kheavyhash/kheavyhash.hpp>
#include "kheavyhash_test_vectors.hpp"


// Layer 2: heavy step (matrix multiply + XOR + KHeavyHash).
// rusty-kaspa matrix.rs::test_heavy_hash literal.
TEST(KHeavyHashPipeline, HeavyHashMatchesReference)
{
    kheavyhash::Hash256 const out{ kheavyhash::heavyHash(kheavyhash::kat::HEAVY_TEST_MATRIX, kheavyhash::kat::HEAVY_INPUT) };
    EXPECT_EQ(out, kheavyhash::kat::HEAVY_EXPECTED);
}


// Layer 2: full pipeline incl. matrix generation. Oracle-minted (pycryptodome) vector.
TEST(KHeavyHashPipeline, CalculatePowMatchesReference)
{
    kheavyhash::Hash256 const out{
        kheavyhash::calculatePow(kheavyhash::kat::FP_PRE, kheavyhash::kat::FP_TIMESTAMP, kheavyhash::kat::FP_NONCE)
    };
    EXPECT_EQ(out, kheavyhash::kat::FP_FINAL);
}


// Accept test: pow <= target (little-endian 256-bit).
TEST(KHeavyHashPipeline, MeetsTargetCompare)
{
    EXPECT_TRUE(kheavyhash::meetsTarget(kheavyhash::kat::FP_FINAL, kheavyhash::kat::FP_TARGET_PASS));
    EXPECT_FALSE(kheavyhash::meetsTarget(kheavyhash::kat::FP_FINAL, kheavyhash::kat::FP_TARGET_FAIL));
}
