#pragma once

#include <cstdint>

#include <algo/hash.hpp>


namespace algo
{
    namespace blake3
    {
        // Alephium PoW: out = BLAKE3(BLAKE3(326-byte mining input)), where the input is
        // nonce(24) || headerBlob(302) -- the nonce is the big-endian 8-byte search value
        // followed by 16 zero bytes (exactly what the miner submits) and headerBlob is
        // left-aligned in `header` (word32[0..75]). Device-neutral host hash: the oracle
        // for the OpenCL/CUDA KATs today, and the per-nonce hash a future CPU resolver
        // would call. Implemented on the vendored BLAKE3 reference (algo/blake3/cpu).
        void hashRef(algo::hash3072 header, uint64_t const nonce, algo::hash256& out);

        // Chain/group index used by the acceptance test: digest byte[31] % CHAIN_NUMBER.
        uint32_t chainIndex(algo::hash256 const& digest);
    }
}
