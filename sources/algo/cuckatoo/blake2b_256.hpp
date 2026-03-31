#pragma once

////////////////////////////////////////////////////////////////////////////
// Minimal self-contained BLAKE2b-256 implementation (CPU only).
//
// Used to derive the 4×uint64 SipHash seed for Cuckatoo32:
//   H1 = blake2b256(pre_pow || nonce_le8)
//   H2 = blake2b256(H1)
//   k0..k3 = H2 interpreted as four little-endian uint64_t
//
// Reference: RFC 7693, https://www.blake2.net
////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <cstring>
#include <algorithm>


namespace algo { namespace cuckatoo {

namespace detail {

static constexpr uint64_t B2B_IV[8] = {
    0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
    0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
    0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};

static constexpr uint8_t B2B_SIGMA[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 },
    {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8 },
    { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13 },
    { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9 },
    {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11 },
    {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10 },
    { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5 },
    {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 }
};

inline uint64_t ror64(uint64_t x, int n)
{
    return (x >> n) | (x << (64 - n));
}

inline void blake2b_compress(uint64_t h[8], const uint8_t block[128], uint64_t bytesHashed, bool lastBlock)
{
    uint64_t m[16];
    std::memcpy(m, block, 128);

    uint64_t v[16];
    for (int i = 0; i < 8; ++i) { v[i]     = h[i];       }
    for (int i = 0; i < 8; ++i) { v[i + 8] = B2B_IV[i]; }
    v[12] ^= bytesHashed;   // low 64 bits of the byte counter
    // v[13] ^= 0;           // high 64 bits of counter (always 0 for us)
    if (lastBlock) { v[14] = ~v[14]; }

    for (int r = 0; r < 12; ++r)
    {
        auto G = [&](int a, int b, int c, int d, int xi, int yi)
        {
            v[a] += v[b] + m[B2B_SIGMA[r][xi]];
            v[d]  = ror64(v[d] ^ v[a], 32);
            v[c] += v[d];
            v[b]  = ror64(v[b] ^ v[c], 24);
            v[a] += v[b] + m[B2B_SIGMA[r][yi]];
            v[d]  = ror64(v[d] ^ v[a], 16);
            v[c] += v[d];
            v[b]  = ror64(v[b] ^ v[c], 63);
        };

        G(0,4, 8,12, 0, 1);  G(1,5, 9,13, 2, 3);
        G(2,6,10,14, 4, 5);  G(3,7,11,15, 6, 7);
        G(0,5,10,15, 8, 9);  G(1,6,11,12,10,11);
        G(2,7, 8,13,12,13);  G(3,4, 9,14,14,15);
    }

    for (int i = 0; i < 8; ++i) { h[i] ^= v[i] ^ v[i + 8]; }
}

} // namespace detail


////////////////////////////////////////////////////////////////////////////
/// Compute BLAKE2b-256 (32-byte output, no key) of @p data.
/// Writes 32 bytes into @p out.
////////////////////////////////////////////////////////////////////////////
inline void blake2b256(const uint8_t* data, std::size_t len, uint8_t* out)
{
    uint64_t h[8];
    for (int i = 0; i < 8; ++i) { h[i] = detail::B2B_IV[i]; }
    // XOR parameter block word 0 with (fanout=1, depth=1, outlen=32)
    //   byte layout: [outlen, keylen, fanout, depth, ...]
    //   = 0x00 00 00 00 00 01 01 20 (little-endian)
    //   = 0x0000000001010020
    h[0] ^= 0x0000000001010020ULL;

    uint8_t  block[128];
    uint64_t bytesHashed = 0ULL;

    if (len == 0)
    {
        std::memset(block, 0, 128);
        detail::blake2b_compress(h, block, 0ULL, true);
    }
    else
    {
        std::size_t pos = 0;
        while (pos < len)
        {
            std::size_t remaining = len - pos;
            bool        isLast    = (remaining <= 128u);
            std::size_t blockLen  = isLast ? remaining : 128u;

            std::memset(block, 0, 128);
            std::memcpy(block, data + pos, blockLen);

            bytesHashed += static_cast<uint64_t>(blockLen);
            detail::blake2b_compress(h, block, bytesHashed, isLast);
            pos += blockLen;
        }
    }

    std::memcpy(out, h, 32u);
}

}} // namespace algo::cuckatoo
