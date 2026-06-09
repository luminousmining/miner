#include <cstring>

#include <gtest/gtest.h>

#include <algo/blake3/blake3_pow.hpp>
#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>


namespace
{
    // 302-byte header from sources/resolver/nvidia/tests/blake3.cpp; toHash<LEFT>
    // right-zero-pads to 384 bytes, the kernel hashes bytes 0..325.
    constexpr char const* HEADER_HEX{
        "000700000000000022d30e3358af8cd1a732e46d47254bb81ffa43cf402dfe001cf5000000000001bd38686272dfd4b55c3559391a"
        "dfab12413c197fa92729465aea000000000001ea959d9fbdd9abeda14a65c0040bb7e626d7c13a6e51979a434f000000000002776a"
        "5133a5941c8bf35e38043a1f4c5700c0844cb835c414c4300000000000017266826c86467877ca053f0776c56a9d340f634237a91e"
        "3865410000000000009dc9c87ce69910b950d0967c9a37930fcfd34f510fc0a014861200000000000157c2d1db2a2af17d3e7560bf"
        "c78b270d2b7e65a7fcff312b3b8310249f0949d3463117e80375771047ab3309102f365ddc609b36eaae1363dfceb25811b3902af4"
        "dd41490d75b7f79a0478f7f721681c59178a2564b84561559a0000018ea81c4bfa1b029ed6" };

    constexpr uint64_t NONCE{ 0x914544566c9a0a4dull };

    // Independent oracle: BLAKE3(BLAKE3(nonce(24) || headerBlob(302))) computed with
    // the reference `blake3` Python package (non-circular KAT). nonce(24) = big-endian
    // 8-byte value + 16 zero bytes; this is the real Alephium PoW layout.
    constexpr char const* EXPECTED_HEX{ "394696ad2377a8ce8525032656e819183c0585d818ff1cffb52aca6acde2d095" };
}


TEST(Blake3Ref, MatchesIndependentDoubleBlake3)
{
    algo::hash3072 const header{ algo::toHash<algo::hash3072>(HEADER_HEX, algo::HASH_SHIFT::LEFT) };
    algo::hash256        out{};

    algo::blake3::hashRef(header, NONCE, out);

    algo::hash256 const expected{ algo::toHash256(EXPECTED_HEX) };
    EXPECT_EQ(0, std::memcmp(out.ubytes, expected.ubytes, 32)) << "got " << algo::toHex(out);

    // digest byte[31] = 0xA5 => chainIndex 5 => fromGroup 1, toGroup 1.
    EXPECT_EQ(5u, algo::blake3::chainIndex(out));
}
