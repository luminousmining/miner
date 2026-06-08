#include <algo/kheavyhash/hashers.hpp>
#include <algo/kheavyhash/keccak.hpp>


namespace kheavyhash
{
    namespace
    {
        // Precomputed cSHAKE256 states (domain + trailing pad baked in), copied
        // verbatim from rusty-kaspa crypto/hashes/src/pow_hashers.rs.
        // POW_INITIAL_STATE = cSHAKE256("ProofOfWorkHash").
        constexpr uint64_t POW_INITIAL_STATE[25]{
            1242148031264380989ull, 3008272977830772284ull, 2188519011337848018ull, 1992179434288343456ull,
            8876506674959887717ull, 5399642050693751366ull, 1745875063082670864ull, 8605242046444978844ull,
            17936695144567157056ull, 3343109343542796272ull, 1123092876221303306ull, 4963925045340115282ull,
            17037383077651887893ull, 16629644495023626889ull, 12833675776649114147ull, 3784524041015224902ull,
            1082795874807940378ull, 13952716920571277634ull, 13411128033953605860ull, 15060696040649351053ull,
            9928834659948351306ull, 5237849264682708699ull, 12825353012139217522ull, 6706187291358897596ull,
            196324915476054915ull,
        };

        // HEAVY_INITIAL_STATE = cSHAKE256("HeavyHash").
        constexpr uint64_t HEAVY_INITIAL_STATE[25]{
            4239941492252378377ull, 8746723911537738262ull, 8796936657246353646ull, 1272090201925444760ull,
            16654558671554924250ull, 8270816933120786537ull, 13907396207649043898ull, 6782861118970774626ull,
            9239690602118867528ull, 11582319943599406348ull, 17596056728278508070ull, 15212962468105129023ull,
            7812475424661425213ull, 3370482334374859748ull, 5690099369266491460ull, 8596393687355028144ull,
            570094237299545110ull, 9119540418498120711ull, 16901969272480492857ull, 13372017233735502424ull,
            14372891883993151831ull, 5171152063242093102ull, 10573107899694386186ull, 6096431547456407061ull,
            1592359455985097269ull,
        };

        inline uint64_t loadLe64(uint8_t const* p)
        {
            uint64_t v{ 0 };
            for (int b{ 0 }; b < 8; ++b)
            {
                v |= static_cast<uint64_t>(p[b]) << (8 * b);
            }
            return v;
        }

        inline Hash256 storeLe256(uint64_t const* state)
        {
            Hash256 out{};
            for (int w{ 0 }; w < 4; ++w)
            {
                for (int b{ 0 }; b < 8; ++b)
                {
                    out[w * 8 + b] = static_cast<uint8_t>((state[w] >> (8 * b)) & 0xFF);
                }
            }
            return out;
        }
    }


    Hash256 powHash(Hash256 const& prePowHash, uint64_t const timestamp, uint64_t const nonce)
    {
        uint64_t state[25];
        for (int i{ 0 }; i < 25; ++i)
        {
            state[i] = POW_INITIAL_STATE[i];
        }
        // message lanes: pre_pow_hash (4 LE words) | timestamp | zero[32] | nonce
        for (int w{ 0 }; w < 4; ++w)
        {
            state[w] ^= loadLe64(prePowHash.data() + w * 8);
        }
        state[4] ^= timestamp;
        state[9] ^= nonce;
        keccakF1600(state);
        return storeLe256(state);
    }


    Hash256 kHeavyHash(Hash256 const& input)
    {
        uint64_t state[25];
        for (int i{ 0 }; i < 25; ++i)
        {
            state[i] = HEAVY_INITIAL_STATE[i];
        }
        for (int w{ 0 }; w < 4; ++w)
        {
            state[w] ^= loadLe64(input.data() + w * 8);
        }
        keccakF1600(state);
        return storeLe256(state);
    }
}
