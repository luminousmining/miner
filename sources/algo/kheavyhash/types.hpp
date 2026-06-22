#pragma once

#include <array>
#include <cstdint>


// Standalone (no-dependency) types for the kHeavyHash CPU reference.
// Kept free of repo/Boost/CUDA headers so the correctness oracle builds and runs
// in a minimal toolchain. GPU/stratum wiring in a later session adapts these to
// the repo's algo::hash256 etc.
namespace algo
{
    namespace kheavyhash
    {
        constexpr std::size_t HASH_SIZE{ 32u };
        constexpr std::size_t MATRIX_DIM{ 64u };

        using Hash256 = std::array<uint8_t, HASH_SIZE>;
        using Matrix = std::array<std::array<uint16_t, MATRIX_DIM>, MATRIX_DIM>;
    }
}
