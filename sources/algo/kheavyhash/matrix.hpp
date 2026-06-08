#pragma once

#include <algo/kheavyhash/types.hpp>


namespace kheavyhash
{
    // Rank of the 64x64 matrix over the reals (float Gaussian elimination, EPS 1e-9),
    // matching rusty-kaspa matrix.rs::compute_rank. Kaspa rejects matrices with rank < 64.
    int computeRank(Matrix const& matrix);

    // Build the 64x64 nibble matrix from a 32-byte seed (pre_pow_hash) via xoshiro256++,
    // regenerating (continuing the same stream) until rank == 64.
    Matrix generateMatrix(Hash256 const& seed);
}
