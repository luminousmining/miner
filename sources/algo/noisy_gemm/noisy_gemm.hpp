#pragma once

#include <array>
#include <cstdint>


namespace algo::noisy_gemm
{
    // Pearl NoisyGEMM protocol — valid rank values for parameter r
    // k must satisfy: 16r <= k <= 4r^2, k <= 65536, 64|k
    static constexpr uint32_t VALID_RANKS[]{ 32u, 64u, 128u, 256u, 512u, 1024u };

    // Maximum winning tiles buffer per kernel launch
    static constexpr uint32_t MAX_WINNING_TILES{ 16u };

    // BLAKE3 domain separation flags used for the PoW check
    static constexpr uint32_t BLAKE3_FLAG_CHUNK_START{ 1u };
    static constexpr uint32_t BLAKE3_FLAG_CHUNK_END  { 2u };
    static constexpr uint32_t BLAKE3_FLAG_ROOT       { 8u };
    static constexpr uint32_t BLAKE3_FLAG_KEYED_HASH { 16u };

    // Combined flags for a single-block keyed hash (CHUNK_START | CHUNK_END | ROOT | KEYED_HASH)
    static constexpr uint32_t BLAKE3_FLAGS_SINGLE_BLOCK_KEYED{ 27u };

    // BLAKE3 IV — fractional parts of square roots of first 8 primes (same as SHA-256)
    static constexpr uint32_t BLAKE3_IV[8]
    {
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
    };

    // BLAKE3 message word schedule — 7 rounds x 16 indices
    static constexpr uint8_t BLAKE3_MSG_SCHEDULE[7][16]
    {
        {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
        {  2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8 },
        {  3,  4, 10, 12, 13,  1,  5, 14,  6,  7,  9,  0, 11, 15,  8,  2 },
        { 10,  7, 12,  9, 14, 11,  6,  2,  0,  4, 15,  8, 13,  3,  5,  1 },
        {  4,  2,  9,  6,  8,  3, 11,  5, 14, 15, 10,  7,  1,  0, 12, 13 },
        { 15, 12,  2, 10,  0,  1, 14,  7,  3,  5,  6,  8, 13,  4, 11,  9 },
        { 14,  3,  6,  8, 13,  7,  1,  9, 11, 10, 15, 12,  5,  0,  4,  2 },
    };

    // Winning tile result — trivially copyable, safe to use on both CPU and GPU
    struct WinningTileGpu
    {
        uint32_t tile_i{};
        uint32_t tile_j{};
        uint32_t M[16]{};
        uint8_t  M_hash[32]{};
    };

    // Mining configuration (protocol parameters mu)
    struct MiningConfig
    {
        uint32_t m{ 4096u };          // rows of A
        uint32_t n{ 4096u };          // columns of B
        uint32_t k{ 256u };           // common dimension (64|k required)
        uint32_t r{ 64u };            // noise rank, in {32,64,128,256,512,1024}
        uint32_t tm{ 16u };           // tile height
        uint32_t tn{ 16u };           // tile width
        double   difficultyBits{ 1.0 }; // b — difficulty target in bits (may be fractional)
    };
}
