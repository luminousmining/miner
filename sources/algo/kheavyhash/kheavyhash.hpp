#pragma once

#include <algo/kheavyhash/types.hpp>


namespace kheavyhash
{
    // Heavy step: expand hash1 to 64 nibbles, matrix-multiply (>>10 to a nibble),
    // XOR with hash1, then KHeavyHash. rusty-kaspa matrix.rs::heavy_hash.
    Hash256 heavyHash(Matrix const& matrix, Hash256 const& hash1);

    // Full per-nonce PoW: generate matrix from pre_pow_hash, PowHash, heavy step.
    // Returns the 32-byte little-endian PoW value. rusty-kaspa lib.rs::calculate_pow.
    Hash256 calculatePow(Hash256 const& prePowHash, uint64_t timestamp, uint64_t nonce);

    // Accept test: powValue <= target, both interpreted as little-endian 256-bit ints.
    bool meetsTarget(Hash256 const& powValueLe, Hash256 const& targetLe);
}
