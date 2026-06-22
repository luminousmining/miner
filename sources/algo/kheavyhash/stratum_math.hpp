#pragma once

#include <cstdint>

#include <algo/kheavyhash/types.hpp>


namespace algo
{
    namespace kheavyhash
    {
        Hash256 prePowFromWords(uint64_t const words[4]);
        Hash256 difficultyToTargetLe(double diff);
    }
}
