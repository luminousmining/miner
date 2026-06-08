#include <gtest/gtest.h>

#include <algo/kheavyhash/matrix.hpp>
#include "kheavyhash_test_vectors.hpp"


namespace
{
    kheavyhash::Matrix toMatrix(std::array<std::array<uint16_t, 64>, 64> const& src)
    {
        kheavyhash::Matrix m{};
        for (size_t i{ 0 }; i < 64; ++i)
        {
            for (size_t j{ 0 }; j < 64; ++j)
            {
                m[i][j] = src[i][j];
            }
        }
        return m;
    }
}


// Layer 1: full-rank gate. rusty-kaspa matrix.rs::test_compute_rank.
TEST(KHeavyHashMatrix, FullRankMatrixHasRank64)
{
    kheavyhash::Matrix const m{ toMatrix(kheavyhash::kat::GEN_EXPECTED_MATRIX) };
    EXPECT_EQ(kheavyhash::computeRank(m), 64);
}


TEST(KHeavyHashMatrix, DuplicatedRowDropsRank)
{
    kheavyhash::Matrix m{ toMatrix(kheavyhash::kat::GEN_EXPECTED_MATRIX) };
    m[0] = m[1];
    EXPECT_EQ(kheavyhash::computeRank(m), 63);
}


// Layer 1: xoshiro256++ seeding + nibble fill + regenerate-until-rank-64.
// rusty-kaspa matrix.rs::test_generate_matrix (seed = [42; 32]).
TEST(KHeavyHashMatrix, GenerateMatchesReference)
{
    kheavyhash::Matrix const expected{ toMatrix(kheavyhash::kat::GEN_EXPECTED_MATRIX) };
    kheavyhash::Matrix const actual{ kheavyhash::generateMatrix(kheavyhash::kat::GEN_SEED) };
    EXPECT_EQ(actual, expected);
}
