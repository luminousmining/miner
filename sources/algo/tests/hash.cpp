#include <string>

#include <gtest/gtest.h>

#include <algo/hash_utils.hpp>


struct HashTest : public testing::Test
{
    HashTest() = default;
    ~HashTest() = default;
};


TEST_F(HashTest, doubleToHash256)
{
    algo::hash256 const hash256{ algo::toHash256(1.999969) };

    // "80000206012bb4b28a376b73d0f6f86aad62b60d19d23efa40b507f1"
    EXPECT_EQ(0x00, hash256.ubytes[0]);
    EXPECT_EQ(0x00, hash256.ubytes[1]);
    EXPECT_EQ(0x00, hash256.ubytes[2]);
    EXPECT_EQ(0x00, hash256.ubytes[3]);
    EXPECT_EQ(0x80, hash256.ubytes[4]);
    EXPECT_EQ(0x00, hash256.ubytes[5]);
    EXPECT_EQ(0x02, hash256.ubytes[6]);
    EXPECT_EQ(0x06, hash256.ubytes[7]);
    EXPECT_EQ(0x01, hash256.ubytes[8]);
    EXPECT_EQ(0x2b, hash256.ubytes[9]);
    EXPECT_EQ(0xb4, hash256.ubytes[10]);
    EXPECT_EQ(0xb2, hash256.ubytes[11]);
    EXPECT_EQ(0x8a, hash256.ubytes[12]);
    EXPECT_EQ(0x37, hash256.ubytes[13]);
    EXPECT_EQ(0x6b, hash256.ubytes[14]);
    EXPECT_EQ(0x73, hash256.ubytes[15]);
    EXPECT_EQ(0xd0, hash256.ubytes[16]);
    EXPECT_EQ(0xf6, hash256.ubytes[17]);
    EXPECT_EQ(0xf8, hash256.ubytes[18]);
    EXPECT_EQ(0x6a, hash256.ubytes[19]);
    EXPECT_EQ(0xad, hash256.ubytes[20]);
    EXPECT_EQ(0x62, hash256.ubytes[21]);
    EXPECT_EQ(0xb6, hash256.ubytes[22]);
    EXPECT_EQ(0x0d, hash256.ubytes[23]);
    EXPECT_EQ(0x19, hash256.ubytes[24]);
    EXPECT_EQ(0xd2, hash256.ubytes[25]);
    EXPECT_EQ(0x3e, hash256.ubytes[26]);
    EXPECT_EQ(0xfa, hash256.ubytes[27]);
    EXPECT_EQ(0x40, hash256.ubytes[28]);
    EXPECT_EQ(0xb5, hash256.ubytes[29]);
    EXPECT_EQ(0x07, hash256.ubytes[30]);
    EXPECT_EQ(0xf1, hash256.ubytes[31]);
}


TEST_F(HashTest, doubleToHash256MinimunValue)
{
    algo::hash256 const hash256{ algo::toHash256(0.0001) };

    EXPECT_EQ(0x00, hash256.ubytes[0]);
    EXPECT_EQ(0x00, hash256.ubytes[1]);
    EXPECT_EQ(0x27, hash256.ubytes[2]);
    EXPECT_EQ(0x0f, hash256.ubytes[3]);
    EXPECT_EQ(0xd8, hash256.ubytes[4]);
    EXPECT_EQ(0xf0, hash256.ubytes[5]);
    EXPECT_EQ(0x00, hash256.ubytes[6]);
    EXPECT_EQ(0x00, hash256.ubytes[7]);
    EXPECT_EQ(0x00, hash256.ubytes[8]);
    EXPECT_EQ(0x00, hash256.ubytes[9]);
    EXPECT_EQ(0x00, hash256.ubytes[10]);
    EXPECT_EQ(0x00, hash256.ubytes[11]);
    EXPECT_EQ(0x00, hash256.ubytes[12]);
    EXPECT_EQ(0x00, hash256.ubytes[13]);
    EXPECT_EQ(0x00, hash256.ubytes[14]);
    EXPECT_EQ(0x00, hash256.ubytes[15]);
    EXPECT_EQ(0x00, hash256.ubytes[16]);
    EXPECT_EQ(0x00, hash256.ubytes[17]);
    EXPECT_EQ(0x00, hash256.ubytes[18]);
    EXPECT_EQ(0x00, hash256.ubytes[19]);
    EXPECT_EQ(0x00, hash256.ubytes[20]);
    EXPECT_EQ(0x00, hash256.ubytes[21]);
    EXPECT_EQ(0x00, hash256.ubytes[22]);
    EXPECT_EQ(0x00, hash256.ubytes[23]);
    EXPECT_EQ(0x00, hash256.ubytes[24]);
    EXPECT_EQ(0x00, hash256.ubytes[25]);
    EXPECT_EQ(0x00, hash256.ubytes[26]);
    EXPECT_EQ(0x00, hash256.ubytes[27]);
    EXPECT_EQ(0x00, hash256.ubytes[28]);
    EXPECT_EQ(0x00, hash256.ubytes[29]);
    EXPECT_EQ(0x00, hash256.ubytes[30]);
    EXPECT_EQ(0x00, hash256.ubytes[31]);
}


TEST_F(HashTest, doubleToHash256MaxValue)
{
    algo::hash256 const hash256{ algo::toHash256(2.0) };

    EXPECT_EQ(0x00, hash256.ubytes[0]);
    EXPECT_EQ(0x00, hash256.ubytes[1]);
    EXPECT_EQ(0x00, hash256.ubytes[2]);
    EXPECT_EQ(0x00, hash256.ubytes[3]);
    EXPECT_EQ(0x7f, hash256.ubytes[4]);
    EXPECT_EQ(0xff, hash256.ubytes[5]);
    EXPECT_EQ(0x80, hash256.ubytes[6]);
    EXPECT_EQ(0x00, hash256.ubytes[7]);
    EXPECT_EQ(0x00, hash256.ubytes[8]);
    EXPECT_EQ(0x00, hash256.ubytes[9]);
    EXPECT_EQ(0x00, hash256.ubytes[10]);
    EXPECT_EQ(0x00, hash256.ubytes[11]);
    EXPECT_EQ(0x00, hash256.ubytes[12]);
    EXPECT_EQ(0x00, hash256.ubytes[13]);
    EXPECT_EQ(0x00, hash256.ubytes[14]);
    EXPECT_EQ(0x00, hash256.ubytes[15]);
    EXPECT_EQ(0x00, hash256.ubytes[16]);
    EXPECT_EQ(0x00, hash256.ubytes[17]);
    EXPECT_EQ(0x00, hash256.ubytes[18]);
    EXPECT_EQ(0x00, hash256.ubytes[19]);
    EXPECT_EQ(0x00, hash256.ubytes[20]);
    EXPECT_EQ(0x00, hash256.ubytes[21]);
    EXPECT_EQ(0x00, hash256.ubytes[22]);
    EXPECT_EQ(0x00, hash256.ubytes[23]);
    EXPECT_EQ(0x00, hash256.ubytes[24]);
    EXPECT_EQ(0x00, hash256.ubytes[25]);
    EXPECT_EQ(0x00, hash256.ubytes[26]);
    EXPECT_EQ(0x00, hash256.ubytes[27]);
    EXPECT_EQ(0x00, hash256.ubytes[28]);
    EXPECT_EQ(0x00, hash256.ubytes[29]);
    EXPECT_EQ(0x00, hash256.ubytes[30]);
    EXPECT_EQ(0x00, hash256.ubytes[31]);
}


TEST_F(HashTest, ShiftingRight)
{
    using namespace std::string_literals;

    algo::hash3072 const original{ algo::toHash<algo::hash3072>(
        "000700000000000022d30e3358af8cd1a732e46d47254bb81ffa43cf402dfe001cf5000000000001bd38686272dfd4b55c3559391adfab"
        "12413c197fa92729465aea000000000001ea959d9fbdd9abeda14a65c0040bb7e626d7c13a6e51979a434f000000000002776a5133a594"
        "1c8bf35e38043a1f4c5700c0844cb835c414c4300000000000017266826c86467877ca053f0776c56a9d340f634237a91e386541000000"
        "0000009dc9c87ce69910b950d0967c9a37930fcfd34f510fc0a014861200000000000157c2d1db2a2af17d3e7560bfc78b270d2b7e65a7"
        "fcff312b3b8310249f0949d3463117e80375771047ab3309102f365ddc609b36eaae1363dfceb25811b3902af4dd41490d75b7f79a0478"
        "f7f721681c59178a2564b84561559a0000018ea81c4bfa1b029ed600000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        "0") };

    algo::hash3072 const shifted{ algo::toHash<algo::hash3072>(
        "000700000000000022d30e3358af8cd1a732e46d47254bb81ffa43cf402dfe001cf5000000000001bd38686272dfd4b55c3559391adfab"
        "12413c197fa92729465aea000000000001ea959d9fbdd9abeda14a65c0040bb7e626d7c13a6e51979a434f000000000002776a5133a594"
        "1c8bf35e38043a1f4c5700c0844cb835c414c4300000000000017266826c86467877ca053f0776c56a9d340f634237a91e386541000000"
        "0000009dc9c87ce69910b950d0967c9a37930fcfd34f510fc0a014861200000000000157c2d1db2a2af17d3e7560bfc78b270d2b7e65a7"
        "fcff312b3b8310249f0949d3463117e80375771047ab3309102f365ddc609b36eaae1363dfceb25811b3902af4dd41490d75b7f79a0478"
        "f7f721681c59178a2564b84561559a0000018ea81c4bfa1b029ed6",
        algo::HASH_SHIFT::LEFT) };

    for (uint32_t i{ 0u }; i < algo::LEN_HASH_3072_WORD_8; ++i)
    {
        ASSERT_EQ(original.ubytes[i], shifted.ubytes[i]) << "index " << i;
    }
}


TEST_F(HashTest, Hash256)
{
    auto const hash_1
    {
        algo::toHash<algo::hash256>(
            "6f109ba5226d1e0814cdeec79f1231d1d48196b5979a6d816e3621a1ef47ad80")
    };
    auto const hash_2
    {
        algo::toHash256(
            "6f109ba5226d1e0814cdeec79f1231d1d48196b5979a6d816e3621a1ef47ad80")
    };

    for (uint64_t i{ 0ull }; i < algo::LEN_HASH_256_WORD_8; ++i)
    {
        ASSERT_EQ(hash_1.ubytes[i], hash_2.ubytes[i]) << "index " << i;
    }
}


TEST_F(HashTest, Hash1024)
{
    auto const hash_1{ algo::toHash<algo::hash1024>(
        "6f109ba5226d1e0814cdeec79f1231d1d48196b5979a6d816e3621a1ef47ad80") };
    auto const hash_2{ algo::toHash1024("6f109ba5226d1e0814cdeec79f1231d1d48196b5979a6d816e3621a1ef47ad80") };

    for (uint64_t i{ 0ull }; i < algo::LEN_HASH_1024_WORD_8; ++i)
    {
        ASSERT_EQ(hash_1.ubytes[i], hash_2.ubytes[i]) << "index " << i;
    }
}


TEST_F(HashTest, Hash256OverlongInputBounded)
{
    // Regression guard: a malicious or MITM'd pool can send a header/seed/boundary
    // field longer than 64 hex chars. toHash256() must stay inside the 32-byte
    // buffer -- no out-of-bounds write / crash -- dropping the excess high bytes
    // and keeping the low 32 bytes (i.e. the last 64 hex chars).
    std::string const canonical{ "6f109ba5226d1e0814cdeec79f1231d1d48196b5979a6d816e3621a1ef47ad80" };
    std::string const overlong{ "ffffffffffffffffdeadbeefdeadbeef" + canonical };

    algo::hash256 const expected{ algo::toHash256(canonical) };
    algo::hash256 const actual{ algo::toHash256(overlong) };

    for (uint64_t i{ 0ull }; i < algo::LEN_HASH_256_WORD_8; ++i)
    {
        ASSERT_EQ(expected.ubytes[i], actual.ubytes[i]) << "index " << i;
    }
}


////////////////////////////////////////////////////////////////////////////////
// decimalToHash<hash512> turns a decimal boundary string into a hash512 whose
// bytes are the ASCII uppercase-hex digits of the value, big-endian (most
// significant digit at ubytes[0]). The input is pool-controlled, so these tests
// pin both the happy path and the two attack shapes the overflow fix guards
// against (see fix/decimaltohash-overflow).
static std::string decimalHashToHex(algo::hash512 const& hash)
{
    std::string str;
    str.reserve(sizeof(algo::hash512));
    for (uint32_t i{ 0u }; i < sizeof(algo::hash512); ++i)
    {
        str.push_back(cast8(hash.ubytes[i]));
    }
    return str;
}


TEST_F(HashTest, decimalToHashValidBoundary)
{
    // Small values verified by hand plus the real 256-bit Autolykos boundary
    // used by the autolykos_v2 stratum/resolver tests. Correct conversion must
    // be preserved by the input-validation fix.
    EXPECT_EQ("00000000000000000000000000000000000000000000000000000000000000FF",
              decimalHashToHex(algo::decimalToHash<algo::hash512>("255")));

    EXPECT_EQ("00000000000000000000000000000000000000000000000000000000499602D2",
              decimalHashToHex(algo::decimalToHash<algo::hash512>("1234567890")));

    EXPECT_EQ("0000000112E0BE826D694B2E62D01511F12A60609E6058143D609A1A1444A664",
              decimalHashToHex(algo::decimalToHash<algo::hash512>(
                  "28948022309329048855892746252171976963209391069768726095651290785380")));
}


TEST_F(HashTest, decimalToHashRejectsNonDecimal)
{
    // A non-digit byte (here a letter) must be rejected cleanly, returning the
    // zero hash instead of feeding a garbage limb into the carry arrays.
    algo::hash512 const fromLetter{ algo::decimalToHash<algo::hash512>("12345x67890") };
    EXPECT_TRUE(algo::isHashEmpty(fromLetter));

    // A high-bit byte is negative as a signed char, so `input[i] - '0'` would
    // otherwise produce a huge limb; it must be rejected the same way.
    std::string highBit{ "1234" };
    highBit.push_back(cast8(0xFF));
    algo::hash512 const fromHighBit{ algo::decimalToHash<algo::hash512>(highBit) };
    EXPECT_TRUE(algo::isHashEmpty(fromHighBit));
}


TEST_F(HashTest, decimalToHashOversizedIsBounded)
{
    // A boundary far larger than the 256-bit field (100 decimal digits ~= 332
    // bits). An oversized number must be handled without corrupting the stack:
    // the carry index is bounded to LIMBS so the high limbs are dropped instead
    // of being written past the fixed-size `accs`/`ts` arrays. The call must
    // return cleanly and the least-significant nibble must still be correct
    // (a run of 100 nines is == 15 mod 16).
    std::string const oversized(100u, '9');
    algo::hash512 const hash{ algo::decimalToHash<algo::hash512>(oversized) };

    EXPECT_EQ('F', cast8(hash.ubytes[sizeof(algo::hash512) - 1u]));
}
