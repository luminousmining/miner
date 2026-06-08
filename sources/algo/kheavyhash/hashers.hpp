#pragma once

#include <algo/kheavyhash/types.hpp>


namespace kheavyhash
{
    // hash1 = cSHAKE256("ProofOfWorkHash") over
    //   pre_pow_hash[32] || timestamp_u64_LE || zero[32] || nonce_u64_LE
    // (rusty-kaspa pow_hashers.rs::PowHash). Output is 32 little-endian bytes.
    Hash256 powHash(Hash256 const& prePowHash, uint64_t timestamp, uint64_t nonce);

    // hash2 step = cSHAKE256("HeavyHash") over 32 bytes (pow_hashers.rs::KHeavyHash).
    Hash256 kHeavyHash(Hash256 const& input);
}
