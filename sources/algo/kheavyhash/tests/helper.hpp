#pragma once

#include <cstdint>

#include <algo/bitwise.hpp>
#include <algo/keccak.hpp>
#include <algo/kheavyhash/types.hpp>


// CPU reference oracle for kHeavyHash, used only by the unit tests to validate the
// GPU (CUDA/OpenCL) kernels.
// Header-only: not part of the production build.
namespace algo
{
    namespace kheavyhash
    {
        // Precomputed cSHAKE256 states (domain + trailing pad baked in), copied
        // verbatim from rusty-kaspa crypto/hashes/src/pow_hashers.rs.
        // POW_INITIAL_STATE = cSHAKE256("ProofOfWorkHash").
        constexpr uint64_t POW_INITIAL_STATE[25]{
            1242148031264380989ull,  3008272977830772284ull,  2188519011337848018ull,  1992179434288343456ull,
            8876506674959887717ull,  5399642050693751366ull,  1745875063082670864ull,  8605242046444978844ull,
            17936695144567157056ull, 3343109343542796272ull,  1123092876221303306ull,  4963925045340115282ull,
            17037383077651887893ull, 16629644495023626889ull, 12833675776649114147ull, 3784524041015224902ull,
            1082795874807940378ull,  13952716920571277634ull, 13411128033953605860ull, 15060696040649351053ull,
            9928834659948351306ull,  5237849264682708699ull,  12825353012139217522ull, 6706187291358897596ull,
            196324915476054915ull,
        };

        // HEAVY_INITIAL_STATE = cSHAKE256("HeavyHash").
        constexpr uint64_t HEAVY_INITIAL_STATE[25]{
            4239941492252378377ull,  8746723911537738262ull,  8796936657246353646ull,  1272090201925444760ull,
            16654558671554924250ull, 8270816933120786537ull,  13907396207649043898ull, 6782861118970774626ull,
            9239690602118867528ull,  11582319943599406348ull, 17596056728278508070ull, 15212962468105129023ull,
            7812475424661425213ull,  3370482334374859748ull,  5690099369266491460ull,  8596393687355028144ull,
            570094237299545110ull,   9119540418498120711ull,  16901969272480492857ull, 13372017233735502424ull,
            14372891883993151831ull, 5171152063242093102ull,  10573107899694386186ull, 6096431547456407061ull,
            1592359455985097269ull,
        };


        // hash1 = cSHAKE256("ProofOfWorkHash") over
        //   pre_pow_hash[32] || timestamp_u64_LE || zero[32] || nonce_u64_LE
        // (rusty-kaspa pow_hashers.rs::PowHash). Output is 32 little-endian bytes.
        inline Hash256 powHash(Hash256 const& prePowHash, uint64_t const timestamp, uint64_t const nonce)
        {
            uint64_t state[25];
            for (int i{ 0 }; i < 25; ++i)
            {
                state[i] = POW_INITIAL_STATE[i];
            }
            // message lanes: pre_pow_hash (4 LE words) | timestamp | zero[32] | nonce
            for (int w{ 0 }; w < 4; ++w)
            {
                state[w] ^= algo::le::uint64(prePowHash.data() + w * 8);
            }
            state[4] ^= timestamp;
            state[9] ^= nonce;
            algo::keccakF1600(state);

            Hash256 out{};
            algo::le::store256(out.data(), state);
            return out;
        }


        // hash2 step = cSHAKE256("HeavyHash") over 32 bytes (pow_hashers.rs::KHeavyHash).
        inline Hash256 kHeavyHash(Hash256 const& input)
        {
            uint64_t state[25];
            for (int i{ 0 }; i < 25; ++i)
            {
                state[i] = HEAVY_INITIAL_STATE[i];
            }
            for (int w{ 0 }; w < 4; ++w)
            {
                state[w] ^= algo::le::uint64(input.data() + w * 8);
            }
            algo::keccakF1600(state);

            Hash256 out{};
            algo::le::store256(out.data(), state);
            return out;
        }


        // Matrix-vector multiply (nibble domain) + XOR + kHeavyHash. The defining
        // "heavy" step of the algorithm.
        inline Hash256 heavyHash(Matrix const& matrix, Hash256 const& hash1)
        {
            // Expand hash1 to 64 nibbles, high nibble first.
            uint16_t vec[64];
            for (int i{ 0 }; i < 32; ++i)
            {
                vec[2 * i] = static_cast<uint16_t>(hash1[i] >> 4);
                vec[2 * i + 1] = static_cast<uint16_t>(hash1[i] & 0x0F);
            }

            // Matrix-vector multiply; two rows collapse to one output byte via >>10 nibbles.
            Hash256 product{};
            for (int i{ 0 }; i < 32; ++i)
            {
                uint16_t sum1{ 0 };
                uint16_t sum2{ 0 };
                for (int j{ 0 }; j < 64; ++j)
                {
                    sum1 = static_cast<uint16_t>(sum1 + matrix[2 * i][j] * vec[j]);
                    sum2 = static_cast<uint16_t>(sum2 + matrix[2 * i + 1][j] * vec[j]);
                }
                product[i] = static_cast<uint8_t>(((sum1 >> 10) << 4) | (sum2 >> 10));
            }

            // XOR with the original hash1 bytes, then KHeavyHash.
            for (int i{ 0 }; i < 32; ++i)
            {
                product[i] ^= hash1[i];
            }
            return kHeavyHash(product);
        }
    }
}
