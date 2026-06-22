#pragma once

#include <algo/kheavyhash/types.hpp>


namespace algo
{
    namespace kheavyhash
    {
        Hash256 heavyHash(Matrix const& matrix, Hash256 const& hash1);
        Hash256 calculatePow(Hash256 const& prePowHash, uint64_t timestamp, uint64_t nonce);
        bool    meetsTarget(Hash256 const& powValueLe, Hash256 const& targetLe);
    }
}
