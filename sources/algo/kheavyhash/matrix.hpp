#pragma once

#include <algo/kheavyhash/types.hpp>


namespace algo
{
    namespace kheavyhash
    {
        int computeRank(Matrix const& matrix);
        Matrix generateMatrix(Hash256 const& seed);
    }
}
