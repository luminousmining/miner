#pragma once

#include <array>
#include <cstdint>


// Standalone (no-dependency) types for the kHeavyHash CPU reference.
// Kept free of repo/Boost/CUDA headers so the correctness oracle builds and runs
// in a minimal toolchain. GPU/stratum wiring in a later session adapts these to
// the repo's algo::hash256 etc.
namespace kheavyhash
{
    using Hash256 = std::array<uint8_t, 32>;
    using Matrix = std::array<std::array<uint16_t, 64>, 64>;
}
