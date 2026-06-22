#include <gtest/gtest.h>

#include <algo/kheavyhash/matrix.hpp>
#include <algo/kheavyhash/tests/helper.hpp>
#include <algo/kheavyhash/tests/kat_vectors.hpp>


// Layer 2: heavy step (matrix multiply + XOR + KHeavyHash).
// rusty-kaspa matrix.rs::test_heavy_hash literal.
TEST(KHeavyHashHeavyHash, heavyHashMatchesReference)
{
    algo::kheavyhash::Hash256 const out{
        algo::kheavyhash::heavyHash(algo::kheavyhash::kat::HEAVY_TEST_MATRIX, algo::kheavyhash::kat::HEAVY_INPUT)
    };
    EXPECT_EQ(out, algo::kheavyhash::kat::HEAVY_EXPECTED);
}


// Layer 2: full pipeline incl. matrix generation. Oracle-minted (pycryptodome) vector.
// The pow pipeline is inlined here: generateMatrix -> powHash -> heavyHash.
TEST(KHeavyHashHeavyHash, calculatePowMatchesReference)
{
    algo::kheavyhash::Matrix const  matrix{ algo::kheavyhash::generateMatrix(algo::kheavyhash::kat::FP_PRE) };
    algo::kheavyhash::Hash256 const hash1{ algo::kheavyhash::powHash(
        algo::kheavyhash::kat::FP_PRE,
        algo::kheavyhash::kat::FP_TIMESTAMP,
        algo::kheavyhash::kat::FP_NONCE) };
    algo::kheavyhash::Hash256 const out{ algo::kheavyhash::heavyHash(matrix, hash1) };
    EXPECT_EQ(out, algo::kheavyhash::kat::FP_FINAL);
}
