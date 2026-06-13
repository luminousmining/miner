#include <cstdint>
#include <cstring>

#include "blake3.h" // vendored reference, via blake3_ref lib

#include <algo/bitwise.hpp>       // bswap32 (native __builtin_bswap32 / _byteswap_ulong)
#include <algo/blake3/blake3.hpp> // CHAIN_NUMBER
#include <algo/blake3/blake3_pow.hpp>
#include <common/cast.hpp> // castU8 / castU32


void algo::blake3::hashRef(algo::hash3072 header, uint64_t const nonce, algo::hash256& out)
{
    // Reconstruct the 326-byte Alephium mining input exactly as the kernel does:
    // word[0..1] = big-endian search value, word[2..5] = 0 (nonce bytes 8..23),
    // word[6..81] = headerBlob (header.word32[0..75]). 326 bytes = 81 little-endian
    // words + the low 2 bytes of word[81].
    uint32_t words[96]{};
    words[0] = bswap32(castU32(nonce >> 32));
    words[1] = bswap32(castU32(nonce & 0xFFFFFFFFu));
    for (uint32_t i{ 0u }; i < 76u; ++i)
    {
        words[6u + i] = header.word32[i];
    }

    uint8_t msg[326]{};
    for (uint32_t i{ 0u }; i < 81u; ++i)
    {
        msg[4u * i + 0u] = castU8(words[i]);
        msg[4u * i + 1u] = castU8(words[i] >> 8);
        msg[4u * i + 2u] = castU8(words[i] >> 16);
        msg[4u * i + 3u] = castU8(words[i] >> 24);
    }
    msg[324] = castU8(words[81]);
    msg[325] = castU8(words[81] >> 8);

    // out = BLAKE3(BLAKE3(msg)). On little-endian hosts the 32 output bytes copied
    // into word32 reproduce the kernel's 8 chaining-value words exactly.
    uint8_t       digest1[32];
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, msg, sizeof(msg));
    blake3_hasher_finalize(&hasher, digest1, sizeof(digest1));

    uint8_t digest2[32];
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, digest1, sizeof(digest1));
    blake3_hasher_finalize(&hasher, digest2, sizeof(digest2));

    std::memcpy(out.word32, digest2, sizeof(digest2));
}


uint32_t algo::blake3::chainIndex(algo::hash256 const& digest)
{
    return (digest.word32[7] >> 24) % algo::blake3::CHAIN_NUMBER;
}
