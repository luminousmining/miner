#pragma once

#include <cstdint>

#include <algo/hash.hpp>


namespace algo
{
    namespace blake3
    {
        // Alephium PoW reference oracle: out = BLAKE3(BLAKE3(326-byte mining input)),
        // where the input is nonce(24) || headerBlob(302) -- the nonce is the
        // big-endian 8-byte search value followed by 16 zero bytes (exactly what the
        // miner submits) and headerBlob is left-aligned in `header` (word32[0..75]).
        // Host oracle for the OpenCL kernel; never on the hot path. Implemented on the
        // vendored BLAKE3 reference (sources/algo/crypto/reference/blake3).
        void hashRef(algo::hash3072 header, uint64_t const nonce, algo::hash256& out);

        // Chain/group index used by the acceptance test: digest byte[31] % CHAIN_NUMBER.
        uint32_t chainIndex(algo::hash256 const& digest);
    }
}
