#include <cmath>

#include <algo/kheavyhash/matrix.hpp>
#include <algo/kheavyhash/xoshiro.hpp>


namespace kheavyhash
{
    int computeRank(Matrix const& matrix)
    {
        // Float Gaussian elimination, mirroring rusty-kaspa matrix.rs::compute_rank.
        constexpr double eps{ 1e-9 };

        double matFloat[MATRIX_DIM][MATRIX_DIM];
        for (size_t i{ 0 }; i < MATRIX_DIM; ++i)
        {
            for (size_t j{ 0 }; j < MATRIX_DIM; ++j)
            {
                matFloat[i][j] = static_cast<double>(matrix[i][j]);
            }
        }

        int  rank{ 0 };
        bool rowSelected[MATRIX_DIM]{};
        for (size_t i{ 0 }; i < MATRIX_DIM; ++i)
        {
            size_t j{ 0 };
            while (j < MATRIX_DIM)
            {
                if (false == rowSelected[j] && std::abs(matFloat[j][i]) > eps)
                {
                    break;
                }
                ++j;
            }
            if (j != MATRIX_DIM)
            {
                ++rank;
                rowSelected[j] = true;
                for (size_t p{ i + 1 }; p < MATRIX_DIM; ++p)
                {
                    matFloat[j][p] /= matFloat[j][i];
                }
                for (size_t k{ 0 }; k < MATRIX_DIM; ++k)
                {
                    if (k != j && std::abs(matFloat[k][i]) > eps)
                    {
                        for (size_t p{ i + 1 }; p < MATRIX_DIM; ++p)
                        {
                            matFloat[k][p] -= matFloat[j][p] * matFloat[k][i];
                        }
                    }
                }
            }
        }
        return rank;
    }


    static Matrix randMatrixNoRankCheck(Xoshiro256pp& generator)
    {
        Matrix mat{};
        for (size_t i{ 0 }; i < MATRIX_DIM; ++i)
        {
            uint64_t val{ 0 };
            for (size_t j{ 0 }; j < MATRIX_DIM; ++j)
            {
                if (0 == j % 16)
                {
                    val = generator.next();
                }
                mat[i][j] = static_cast<uint16_t>((val >> (4 * (j % 16))) & 0x0Full);
            }
        }
        return mat;
    }


    Matrix generateMatrix(Hash256 const& seed)
    {
        // Regenerate from the SAME continuing stream until full rank, per matrix.rs::generate.
        // A random nibble matrix is full-rank with overwhelming probability, so this
        // converges in 1-2 draws; the bound only guards against a pathological stream
        // (e.g. a degenerate seed) turning the retry into an infinite loop.
        constexpr int maxAttempts{ 1024 };
        Xoshiro256pp  generator{ seed };
        Matrix        mat{};
        for (int attempt{ 0 }; attempt < maxAttempts; ++attempt)
        {
            mat = randMatrixNoRankCheck(generator);
            if (static_cast<int>(MATRIX_DIM) == computeRank(mat))
            {
                return mat;
            }
        }
        return mat;
    }
}
