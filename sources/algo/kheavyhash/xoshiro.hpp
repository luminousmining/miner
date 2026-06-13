#pragma once

#include <algo/kheavyhash/types.hpp>


namespace kheavyhash
{
    // xoshiro256++ PRNG, seeded from a 32-byte hash read as 4 little-endian u64 words.
    // Mirrors rusty-kaspa xoshiro.rs::XoShiRo256PlusPlus.
    class Xoshiro256pp
    {
      public:
        explicit Xoshiro256pp(Hash256 const& seed)
        {
            for (int i{ 0 }; i < 4; ++i)
            {
                uint64_t word{ 0 };
                for (int b{ 0 }; b < 8; ++b)
                {
                    word |= static_cast<uint64_t>(seed[i * 8 + b]) << (8 * b);
                }
                s[i] = word;
            }
        }

        uint64_t next()
        {
            uint64_t const res{ s[0] + rotl(s[0] + s[3], 23) };
            uint64_t const t{ s[1] << 17 };
            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];
            s[2] ^= t;
            s[3] = rotl(s[3], 45);
            return res;
        }

      private:
        static uint64_t rotl(uint64_t const x, int const k)
        {
            return (x << k) | (x >> (64 - k));
        }

        uint64_t s[4]{};
    };
}
