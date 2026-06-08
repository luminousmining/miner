#include <algo/kheavyhash/hashers.hpp>
#include <algo/kheavyhash/kheavyhash.hpp>
#include <algo/kheavyhash/matrix.hpp>


namespace kheavyhash
{
    Hash256 heavyHash(Matrix const& matrix, Hash256 const& hash1)
    {
        // Expand hash1 to 64 nibbles, high nibble first.
        uint16_t vec[64];
        for (int i{ 0 }; i < 32; ++i)
        {
            vec[2 * i] = static_cast<uint16_t>(hash1[i] >> 4);
            vec[2 * i + 1] = static_cast<uint16_t>(hash1[i] & 0x0F);
        }

        // Matrix-vector multiply; two rows collapse to one output byte via >>10 nibbles.
        Hash256 product{};
        for (int i{ 0 }; i < 32; ++i)
        {
            uint16_t sum1{ 0 };
            uint16_t sum2{ 0 };
            for (int j{ 0 }; j < 64; ++j)
            {
                sum1 = static_cast<uint16_t>(sum1 + matrix[2 * i][j] * vec[j]);
                sum2 = static_cast<uint16_t>(sum2 + matrix[2 * i + 1][j] * vec[j]);
            }
            product[i] = static_cast<uint8_t>(((sum1 >> 10) << 4) | (sum2 >> 10));
        }

        // XOR with the original hash1 bytes, then KHeavyHash.
        for (int i{ 0 }; i < 32; ++i)
        {
            product[i] ^= hash1[i];
        }
        return kHeavyHash(product);
    }


    Hash256 calculatePow(Hash256 const& prePowHash, uint64_t const timestamp, uint64_t const nonce)
    {
        Matrix const  matrix{ generateMatrix(prePowHash) };
        Hash256 const hash1{ powHash(prePowHash, timestamp, nonce) };
        return heavyHash(matrix, hash1);
    }


    bool meetsTarget(Hash256 const& powValueLe, Hash256 const& targetLe)
    {
        // Compare as little-endian 256-bit integers: scan from most-significant byte.
        for (int i{ 31 }; i >= 0; --i)
        {
            if (powValueLe[i] != targetLe[i])
            {
                return powValueLe[i] < targetLe[i];
            }
        }
        return true;  // equal => pow <= target
    }
}
