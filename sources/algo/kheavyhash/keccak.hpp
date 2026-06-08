#pragma once

#include <cstdint>


namespace kheavyhash
{
    // Standard Keccak-f[1600] permutation (24 rounds) on a 25-lane state.
    // Kaspa's PowHash/KHeavyHash absorb their cSHAKE256 message into a precomputed
    // initial state and apply exactly one permutation.
    void keccakF1600(uint64_t* state);
}
