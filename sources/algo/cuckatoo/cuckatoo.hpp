#pragma once

#include <cstdint>


namespace algo
{
    namespace cuckatoo
    {
        ////////////////////////////////////////////////////////////////////////
        // Graph parameters (Cuckatoo32)
        constexpr uint32_t EDGE_BITS{ 32u };
        constexpr uint32_t PROOF_SIZE{ 42u };

        constexpr uint64_t NUM_EDGES{ 1ull << EDGE_BITS }; // 4 294 967 296
        constexpr uint64_t NUM_NODES{ NUM_EDGES };
        constexpr uint64_t EDGE_MASK{ NUM_EDGES - 1ull };
        constexpr uint64_t NODE_MASK{ NUM_NODES - 1ull };

        ////////////////////////////////////////////////////////////////////////
        // Edge-block geometry (used in GPU edge generation)
        constexpr uint32_t EDGE_BLOCK_BITS{ 6u };
        constexpr uint32_t EDGE_BLOCK_SIZE{ 1u << EDGE_BLOCK_BITS }; // 64
        constexpr uint32_t EDGE_BLOCK_MASK{ EDGE_BLOCK_SIZE - 1u };

        ////////////////////////////////////////////////////////////////////////
        // Trimming rounds
        // 96 rounds is standard for Cuckatoo32.
        constexpr uint32_t TRIM_ROUNDS{ 96u };

        ////////////////////////////////////////////////////////////////////////
        // Default GPU occupancy
        constexpr uint32_t DEFAULT_BLOCKS { 128u };
        constexpr uint32_t DEFAULT_THREADS{ 128u };

        ////////////////////////////////////////////////////////////////////////
        // Memory footprint
        // Edge bitmap  : NUM_EDGES / 8 = 512 MB
        // Node counter : NUM_NODES * 1 B = 4 GB  (full lean approach)
        constexpr uint64_t EDGE_BITMAP_BYTES{ NUM_EDGES / 8ull }; // 512 MB

        ////////////////////////////////////////////////////////////////////////
        // Difficulty conversion
        // Grin sends difficulty as a plain uint64.
        // boundary = 2^256 / difficulty  (approximated via algo::toHash256)
        constexpr uint64_t MIN_DIFFICULTY{ 1ull };
    }
}
