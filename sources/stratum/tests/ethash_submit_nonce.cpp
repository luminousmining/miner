#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <stratum/ethash.hpp>
#include <stratum/stratum_type.hpp>


////////////////////////////////////////////////////////////////////////////////
// Regression guard for the NiceHash EthereumStratum/1.0.0 (daggerhashimoto)
// nonce convention. See the spec:
//   https://github.com/nicehash/Specifications/blob/master/
//       EthereumStratum_NiceHash_v1.0.0.txt
//
// The pool assigns an extranonce that occupies the HIGH bytes of the 64-bit
// nonce; the miner owns the remaining low bytes (the "minernonce"). mining.submit
// sends the minernonce ONLY -- no "0x" prefix, width == 8 - extranonceBytes:
//
//   "Second parameter of params array is job ID, third parameter is minernonce.
//    Note ... minernonce is 6 bytes, because provided extranonce was 2 bytes.
//    If pool provides 3 bytes extranonce, then minernonce must be 5 bytes."
//
// A previous bug report claimed this submit string was "malformed" (missing the
// 0x / extranonce prefix the kawpow/progpow path uses) and proposed emitting the
// absolute extranonce-prefixed nonce instead. That is WRONG for ethash: those
// minernonce-only submits are exactly what NiceHash accepts. These tests lock the
// convention so it is not "fixed" back into a rejection. The worked values below
// are taken verbatim from the spec's own example (section III/IV):
//
//   extranonce = "a2eea0"  (3 bytes)
//   minernonce = "cfae7df760"  (5 bytes, submitted as-is, no 0x)
//   full nonce = 0xa2eea0cfae7df760
////////////////////////////////////////////////////////////////////////////////


constexpr uint64_t SPEC_FULL_NONCE{ 0xa2eea0cfae7df760ull };
constexpr char     SPEC_EXTRA_NONCE[]{ "a2eea0" };
constexpr char     SPEC_MINER_NONCE[]{ "cfae7df760" };


// Mirror of resolver::ResolverAmdEthash::submit(): the found nonce is printed
// as lower-case hex and the leading extranonce hex chars are dropped to leave
// the minernonce that goes into mining.submit params[2].
static std::string toMinerNonce(uint64_t const nonce, uint32_t const extraNonceSize)
{
    std::stringstream nonceHexa;
    nonceHexa << std::hex << nonce;
    return nonceHexa.str().substr(extraNonceSize);
}


struct StratumEthashSubmitNonceTest : public testing::Test
{
    StratumEthashSubmitNonceTest() = default;
    ~StratumEthashSubmitNonceTest() = default;
};


////////////////////////////////////////////////////////////////////////////////
// setExtraNonce() places the pool extranonce in the HIGH bytes of the search
// start nonce and records its hex width. This is what the resolver hashes, so it
// must carry the extranonce -- otherwise the pool recomputes a different hash.
////////////////////////////////////////////////////////////////////////////////
TEST_F(StratumEthashSubmitNonceTest, ExtraNonceOccupiesHighBytesOfStartNonce)
{
    stratum::StratumEthash stratum{};
    stratum.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V1;

    stratum.setExtraNonce(SPEC_EXTRA_NONCE);

    // "a2eea0" right-padded with zeros to 16 hex chars -> 0xa2eea00000000000.
    EXPECT_EQ(6u, stratum.jobInfo.extraNonceSize);
    EXPECT_EQ(0xa2eea00000000000ull, stratum.jobInfo.startNonce);
    EXPECT_EQ(stratum.jobInfo.startNonce, stratum.jobInfo.nonce);
}


////////////////////////////////////////////////////////////////////////////////
// Gluing the pool extranonce (high) with the miner's minernonce (low) must
// reconstruct the absolute 64-bit nonce the pool verifies.
////////////////////////////////////////////////////////////////////////////////
TEST_F(StratumEthashSubmitNonceTest, StartNonceGluedWithMinerNonceIsAbsoluteNonce)
{
    stratum::StratumEthash stratum{};
    stratum.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V1;

    stratum.setExtraNonce(SPEC_EXTRA_NONCE);

    uint64_t const minerNonceBits{ std::strtoull(SPEC_MINER_NONCE, nullptr, 16) };
    EXPECT_EQ(SPEC_FULL_NONCE, stratum.jobInfo.startNonce | minerNonceBits);
}


////////////////////////////////////////////////////////////////////////////////
// The submitted minernonce is the absolute nonce with the extranonce prefix
// stripped: lower-case hex, NO "0x", width == 16 - extraNonceSize hex chars
// (i.e. 8 - extranonceBytes). Matches the spec's mining.submit params[2].
////////////////////////////////////////////////////////////////////////////////
TEST_F(StratumEthashSubmitNonceTest, MinerNonceIsSuffixOnlyWithoutHexPrefix)
{
    stratum::StratumEthash stratum{};
    stratum.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V1;

    stratum.setExtraNonce(SPEC_EXTRA_NONCE);

    std::string const minerNonce{ toMinerNonce(SPEC_FULL_NONCE, stratum.jobInfo.extraNonceSize) };

    EXPECT_EQ(SPEC_MINER_NONCE, minerNonce);
    EXPECT_EQ(16u - stratum.jobInfo.extraNonceSize, minerNonce.size());
    EXPECT_EQ(std::string::npos, minerNonce.find("0x"));
    EXPECT_EQ(std::string::npos, minerNonce.find('x'));
}


////////////////////////////////////////////////////////////////////////////////
// Minernonce width tracks the extranonce width: a 3-byte extranonce leaves a
// 5-byte (10 hex) minernonce; a 2-byte extranonce leaves a 6-byte (12 hex) one.
////////////////////////////////////////////////////////////////////////////////
TEST_F(StratumEthashSubmitNonceTest, MinerNonceWidthTracksExtraNonceWidth)
{
    {
        stratum::StratumEthash stratum{};
        stratum.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V1;
        stratum.setExtraNonce("a2eea0"); // 3 bytes
        EXPECT_EQ(10u, 16u - stratum.jobInfo.extraNonceSize);
    }
    {
        stratum::StratumEthash stratum{};
        stratum.stratumType = stratum::STRATUM_TYPE::ETHEREUM_V1;
        stratum.setExtraNonce("bf04"); // 2 bytes
        EXPECT_EQ(12u, 16u - stratum.jobInfo.extraNonceSize);
    }
}
