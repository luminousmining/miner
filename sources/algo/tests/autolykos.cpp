#include <gtest/gtest.h>

#include <algo/autolykos/autolykos.hpp>
#include <algo/bitwise.hpp>
#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>


// Canonical Autolykos2 proof-of-work vector from the Ergo reference client
// (AutolykosPowSchemeSpec, "test vectors for first increase in N value", height
// 614,400, protocol version 2; algorithm sigma/pow/Autolykos2PowValidation.scala).
// Re-derived independently against Blake2b256:
//   msg    = 548c3e60..7e864f  (Blake2b256 of the header without PoW)
//   nonce  = 0x0000000000003105
//   height = 614400 -> N = calcN = 70464240
//   hit    = 0002fcb113fe65e5754959872dfdbffea0489bf830beb4961ddc0e9e66a1412a
//   b      = 7067388259..301849  (target at difficulty 16384)
// hit < b, so nonce 0x3105 is a valid solution. The CPU re-check used by
// ResolverAmdAutolykosV2::getResultCache to filter GPU candidates before submit
// must therefore accept it. (The matching GPU pipeline is pinned by
// ResolverAutolykosv2AmdTest.acceptsCanonicalErgoVectorHeight614400.)
TEST(AutolykosV2CpuShare, acceptsCanonicalErgoVectorHeight614400)
{
    algo::hash256 header{ algo::toHash256("548c3e602a8f36f8f2738f5f643b02425038044d98543a51cabaa9785e7e864f") };

    // Build the boundary through the same pipeline the live stratum uses
    // (StratumAutolykosV2::onMiningNotify), so isValidShare sees the target in the
    // little-endian layout it expects.
    algo::hash256 boundary{ algo::toHash2<algo::hash256, algo::hash512>(
        algo::toLittleEndian<algo::hash512>(algo::decimalToHash<algo::hash512>(
            "7067388259113537318333190002971674063283542741642755394446115914399301849"))) };

    // isValidShare takes the height already byte-swapped to big-endian, exactly as
    // the resolver passes it (parameters.hostHeight = algo::be::uint32(blockNumber)).
    EXPECT_TRUE(algo::autolykos_v2::mhssamadani::isValidShare(header, boundary, 0x3105ull, algo::be::uint32(614400u)));
}
