#include <gtest/gtest.h>

#include <algo/kheavyhash/matrix.hpp>
#include <algo/kheavyhash/tests/kat_vectors.hpp>


namespace
{
    algo::kheavyhash::Matrix toMatrix(std::array<std::array<uint16_t, 64>, 64> const& src)
    {
        algo::kheavyhash::Matrix m{};
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
    algo::kheavyhash::Matrix const m{ toMatrix(algo::kheavyhash::kat::GEN_EXPECTED_MATRIX) };
    EXPECT_EQ(algo::kheavyhash::computeRank(m), 64);
}


TEST(KHeavyHashMatrix, DuplicatedRowDropsRank)
{
    algo::kheavyhash::Matrix m{ toMatrix(algo::kheavyhash::kat::GEN_EXPECTED_MATRIX) };
    m[0] = m[1];
    EXPECT_EQ(algo::kheavyhash::computeRank(m), 63);
}


// Layer 1: xoshiro256++ seeding + nibble fill + regenerate-until-rank-64.
// rusty-kaspa matrix.rs::test_generate_matrix (seed = [42; 32]).
TEST(KHeavyHashMatrix, GenerateMatchesReference)
{
    algo::kheavyhash::Matrix const expected{ toMatrix(algo::kheavyhash::kat::GEN_EXPECTED_MATRIX) };
    algo::kheavyhash::Matrix const actual{ algo::kheavyhash::generateMatrix(algo::kheavyhash::kat::GEN_SEED) };
    EXPECT_EQ(actual, expected);
}
