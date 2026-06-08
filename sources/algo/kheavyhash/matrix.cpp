#include <cmath>

#include <algo/kheavyhash/matrix.hpp>
#include <algo/kheavyhash/xoshiro.hpp>


namespace kheavyhash
{
    int computeRank(Matrix const& matrix)
    {
        // Float Gaussian elimination, mirroring rusty-kaspa matrix.rs::compute_rank.
        constexpr double eps{ 1e-9 };

        double matFloat[64][64];
        for (size_t i{ 0 }; i < 64; ++i)
        {
            for (size_t j{ 0 }; j < 64; ++j)
            {
                matFloat[i][j] = static_cast<double>(matrix[i][j]);
            }
        }

        int  rank{ 0 };
        bool rowSelected[64]{};
        for (size_t i{ 0 }; i < 64; ++i)
        {
            size_t j{ 0 };
            while (j < 64)
            {
                if (false == rowSelected[j] && std::abs(matFloat[j][i]) > eps)
                {
                    break;
                }
                ++j;
            }
            if (j != 64)
            {
                ++rank;
                rowSelected[j] = true;
                for (size_t p{ i + 1 }; p < 64; ++p)
                {
                    matFloat[j][p] /= matFloat[j][i];
                }
                for (size_t k{ 0 }; k < 64; ++k)
                {
                    if (k != j && std::abs(matFloat[k][i]) > eps)
                    {
                        for (size_t p{ i + 1 }; p < 64; ++p)
                        {
                            matFloat[k][p] -= matFloat[j][p] * matFloat[k][i];
                        }
                    }
                }
            }
        }
        return rank;
    }


    namespace
    {
        Matrix randMatrixNoRankCheck(Xoshiro256pp& generator)
        {
            Matrix mat{};
            for (size_t i{ 0 }; i < 64; ++i)
            {
                uint64_t val{ 0 };
                for (size_t j{ 0 }; j < 64; ++j)
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
    }


    Matrix generateMatrix(Hash256 const& seed)
    {
        // Regenerate from the SAME continuing stream until full rank, per matrix.rs::generate.
        Xoshiro256pp generator{ seed };
        while (true)
        {
            Matrix const mat{ randMatrixNoRankCheck(generator) };
            if (64 == computeRank(mat))
            {
                return mat;
            }
        }
    }
}
