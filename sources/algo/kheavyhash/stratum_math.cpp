#include <cmath>

#include <algo/kheavyhash/stratum_math.hpp>


namespace kheavyhash
{
    Hash256 prePowFromWords(uint64_t const words[4])
    {
        Hash256 out{};
        for (int w{ 0 }; w < 4; ++w)
        {
            for (int b{ 0 }; b < 8; ++b)
            {
                out[w * 8 + b] = static_cast<uint8_t>((words[w] >> (8 * b)) & 0xFF);
            }
        }
        return out;
    }


    Hash256 difficultyToTargetLe(double const diff)
    {
        Hash256 out{};
        if (diff <= 0.0)
        {
            return out;  // impossible target
        }

        // Quantise diff to Q16.16 so fractional difficulties are honoured while
        // the divisor stays <= 2^32 (keeps the long-division remainder in 64 bits).
        // Assumes a pool difficulty in (1/65536, 65536), which covers Kaspa testnet.
        double const dScaled{ std::round(diff * 65536.0) };
        if (dScaled < 1.0)
        {
            return out;
        }
        uint64_t divisor{ static_cast<uint64_t>(dScaled) };
        if (divisor > 0xFFFFFFFFull)
        {
            divisor = 0xFFFFFFFFull;
        }

        // numerator = (2^224 - 1) << 16, as 8 little-endian 32-bit limbs.
        uint32_t maxTarget[8]{ 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                               0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u };
        uint32_t num[8]{};
        num[0] = maxTarget[0] << 16;
        for (int i{ 1 }; i < 8; ++i)
        {
            num[i] = static_cast<uint32_t>((maxTarget[i] << 16) | (maxTarget[i - 1] >> 16));
        }

        // quotient = numerator / divisor (schoolbook long division, high limb first).
        uint32_t quotient[8]{};
        uint64_t remainder{ 0ull };
        for (int i{ 7 }; i >= 0; --i)
        {
            uint64_t const cur{ (remainder << 32) | static_cast<uint64_t>(num[i]) };
            quotient[i] = static_cast<uint32_t>(cur / divisor);
            remainder = cur % divisor;
        }

        // serialise quotient little-endian.
        for (int i{ 0 }; i < 8; ++i)
        {
            for (int b{ 0 }; b < 4; ++b)
            {
                out[i * 4 + b] = static_cast<uint8_t>((quotient[i] >> (8 * b)) & 0xFF);
            }
        }
        return out;
    }
}
