#pragma once

#include <cstdint>

#include <algo/kheavyhash/types.hpp>


namespace kheavyhash
{
    // Reconstruct the 32-byte pre-pow hash from the 4 little-endian u64 words a
    // Kaspa pool sends in mining.notify. Each word is written back little-endian:
    // word w -> bytes[w*8 .. w*8+7]. This is the inverse of the powHash word load.
    Hash256 prePowFromWords(uint64_t const words[4]);

    // Convert a stratum difficulty into the 256-bit target the kernel compares
    // against (little-endian: out[0] is the least-significant byte).
    //
    //   maxTarget = 2^224 - 1     (kaspa-stratum-bridge hasher.go; 56 hex F's)
    //   target    = floor(maxTarget / diff)
    //
    // Computed with exact integer long-division (diff is quantised to Q16.16, so
    // fractional difficulties down to 1/65536 are honoured). NOTE: the maxTarget
    // constant and any per-pool scaling are the single most pool-specific value in
    // the protocol and MUST be confirmed against the pool chosen for testing
    // before trusting share acceptance (see NOTES.md s4.3).
    Hash256 difficultyToTargetLe(double diff);
}
