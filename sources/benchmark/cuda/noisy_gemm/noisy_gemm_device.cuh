#pragma once

// BLAKE3 device implementation + inline PoW check.
// Used by naive NoisyGEMM kernels (benchmark Phase A).
// Design: one thread per tile, no inter-thread collaboration.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algo/noisy_gemm/noisy_gemm.hpp>


namespace noisy_gemm::device
{
    // WinningTileGpu is defined in algo/noisy_gemm/noisy_gemm.hpp (no CUDA dependency)
    using WinningTileGpu = algo::noisy_gemm::WinningTileGpu;


    // =========================================================================
    // BLAKE3 device — conformant implementation
    // Reference: https://github.com/BLAKE3-team/BLAKE3-specs/blake3.pdf
    // =========================================================================

    // BLAKE3 IV (same as SHA-256)
    static __device__ __constant__ uint32_t BLAKE3_IV_D[8] =
    {
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
    };

    // Message word permutation schedule per round
    static __device__ __constant__ uint8_t BLAKE3_SCHEDULE_D[7][16] =
    {
        {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
        {  2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8 },
        {  3,  4, 10, 12, 13,  1,  5, 14,  6,  7,  9,  0, 11, 15,  8,  2 },
        { 10,  7, 12,  9, 14, 11,  6,  2,  0,  4, 15,  8, 13,  3,  5,  1 },
        {  4,  2,  9,  6,  8,  3, 11,  5, 14, 15, 10,  7,  1,  0, 12, 13 },
        { 15, 12,  2, 10,  0,  1, 14,  7,  3,  5,  6,  8, 13,  4, 11,  9 },
        { 14,  3,  6,  8, 13,  7,  1,  9, 11, 10, 15, 12,  5,  0,  4,  2 },
    };

    // CHUNK_START | CHUNK_END | ROOT | KEYED_HASH
    static constexpr uint32_t BLAKE3_FLAGS_KEYED{ 27u };


    // BLAKE3 quarter-round G function (32-bit, rotations differ from BLAKE2b)
    __device__ __forceinline__
    void blake3G(
        uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
        uint32_t  mx, uint32_t my)
    {
        a = a + b + mx;
        d = __funnelshift_r(d ^ a, d ^ a, 16u);
        c = c + d;
        b = __funnelshift_r(b ^ c, b ^ c, 12u);
        a = a + b + my;
        d = __funnelshift_r(d ^ a, d ^ a, 8u);
        c = c + d;
        b = __funnelshift_r(b ^ c, b ^ c, 7u);
    }


    // One BLAKE3 round using the per-round message schedule
    __device__ __forceinline__
    void blake3Round(uint32_t state[16], uint32_t const m[16], uint32_t const round)
    {
        uint8_t const* s{ BLAKE3_SCHEDULE_D[round] };
        // Column steps
        blake3G(state[0], state[4], state[8],  state[12], m[s[0]],  m[s[1]]);
        blake3G(state[1], state[5], state[9],  state[13], m[s[2]],  m[s[3]]);
        blake3G(state[2], state[6], state[10], state[14], m[s[4]],  m[s[5]]);
        blake3G(state[3], state[7], state[11], state[15], m[s[6]],  m[s[7]]);
        // Diagonal steps
        blake3G(state[0], state[5], state[10], state[15], m[s[8]],  m[s[9]]);
        blake3G(state[1], state[6], state[11], state[12], m[s[10]], m[s[11]]);
        blake3G(state[2], state[7], state[8],  state[13], m[s[12]], m[s[13]]);
        blake3G(state[3], state[4], state[9],  state[14], m[s[14]], m[s[15]]);
    }


    // Keyed BLAKE3 hash of M[16 uint32] (64 bytes) using a 32-byte key.
    // Writes 32 bytes to output.
    // Single chunk, single block, keyed mode — conformant to the BLAKE3 spec.
    __device__ __forceinline__
    void blake3HashM(
        uint32_t const M[16],
        uint8_t  const key[32],
        uint8_t        output[32])
    {
        // Load key as uint32 little-endian words
        uint32_t cv[8];
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            uint32_t const base{ i * 4u };
            cv[i] = static_cast<uint32_t>(key[base])
                  | (static_cast<uint32_t>(key[base + 1u]) << 8u)
                  | (static_cast<uint32_t>(key[base + 2u]) << 16u)
                  | (static_cast<uint32_t>(key[base + 3u]) << 24u);
        }

        // State: [cv | IV[0..3] | counter_low | counter_high | block_len | flags]
        uint32_t state[16]
        {
            cv[0], cv[1], cv[2], cv[3],
            cv[4], cv[5], cv[6], cv[7],
            BLAKE3_IV_D[0], BLAKE3_IV_D[1], BLAKE3_IV_D[2], BLAKE3_IV_D[3],
            0u,   // counter_low
            0u,   // counter_high
            64u,  // block_len = 64 bytes
            BLAKE3_FLAGS_KEYED
        };

        // 7 mixing rounds
        for (uint32_t r{ 0u }; r < 7u; ++r)
        {
            blake3Round(state, M, r);
        }

        // Finalize: output[i] = state[i] XOR state[i+8]
        for (uint32_t i{ 0u }; i < 8u; ++i)
        {
            uint32_t const word{ state[i] ^ state[i + 8u] };
            uint32_t const base{ i * 4u };
            output[base]      = static_cast<uint8_t>(word);
            output[base + 1u] = static_cast<uint8_t>(word >> 8u);
            output[base + 2u] = static_cast<uint8_t>(word >> 16u);
            output[base + 3u] = static_cast<uint8_t>(word >> 24u);
        }
    }


    // =========================================================================
    // Inline PoW check
    // Returns true when hash[32] <= threshold[4 x uint64] as little-endian uint256.
    // =========================================================================
    __device__ __forceinline__
    bool checkPow(uint8_t const hash[32], uint64_t const threshold[4])
    {
        // Interpret hash as 4 little-endian uint64 words
        uint64_t h[4];
        for (uint32_t i{ 0u }; i < 4u; ++i)
        {
            uint32_t const base{ i * 8u };
            h[i] = static_cast<uint64_t>(hash[base])
                 | (static_cast<uint64_t>(hash[base + 1u]) << 8u)
                 | (static_cast<uint64_t>(hash[base + 2u]) << 16u)
                 | (static_cast<uint64_t>(hash[base + 3u]) << 24u)
                 | (static_cast<uint64_t>(hash[base + 4u]) << 32u)
                 | (static_cast<uint64_t>(hash[base + 5u]) << 40u)
                 | (static_cast<uint64_t>(hash[base + 6u]) << 48u)
                 | (static_cast<uint64_t>(hash[base + 7u]) << 56u);
        }

        // uint256 little-endian comparison: most significant word is at index 3
        for (int32_t i{ 3 }; i >= 0; --i)
        {
            if (h[i] < threshold[i]) { return true;  }
            if (h[i] > threshold[i]) { return false; }
        }
        return true; // exact equality
    }
}
