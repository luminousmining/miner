#include <algo/kheavyhash/keccak.hpp>


namespace kheavyhash
{
    namespace
    {
        constexpr uint64_t ROUND_CONSTANTS[24]{
            0x0000000000000001ull, 0x0000000000008082ull, 0x800000000000808aull, 0x8000000080008000ull,
            0x000000000000808bull, 0x0000000080000001ull, 0x8000000080008081ull, 0x8000000000008009ull,
            0x000000000000008aull, 0x0000000000000088ull, 0x0000000080008009ull, 0x000000008000000aull,
            0x000000008000808bull, 0x800000000000008bull, 0x8000000000008089ull, 0x8000000000008003ull,
            0x8000000000008002ull, 0x8000000000000080ull, 0x000000000000800aull, 0x800000008000000aull,
            0x8000000080008081ull, 0x8000000000008080ull, 0x0000000080000001ull, 0x8000000080008008ull,
        };

        constexpr int ROTATIONS[24]{ 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44 };

        constexpr int PI_LANE[24]{ 10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1 };

        inline uint64_t rotl(uint64_t const x, int const k)
        {
            return (x << k) | (x >> (64 - k));
        }
    }


    void keccakF1600(uint64_t* a)
    {
        for (int round{ 0 }; round < 24; ++round)
        {
            // Theta
            uint64_t bc[5];
            for (int i{ 0 }; i < 5; ++i)
            {
                bc[i] = a[i] ^ a[i + 5] ^ a[i + 10] ^ a[i + 15] ^ a[i + 20];
            }
            for (int i{ 0 }; i < 5; ++i)
            {
                uint64_t const t{ bc[(i + 4) % 5] ^ rotl(bc[(i + 1) % 5], 1) };
                for (int j{ 0 }; j < 25; j += 5)
                {
                    a[j + i] ^= t;
                }
            }

            // Rho + Pi
            uint64_t t{ a[1] };
            for (int i{ 0 }; i < 24; ++i)
            {
                int const j{ PI_LANE[i] };
                uint64_t const tmp{ a[j] };
                a[j] = rotl(t, ROTATIONS[i]);
                t = tmp;
            }

            // Chi
            for (int j{ 0 }; j < 25; j += 5)
            {
                for (int i{ 0 }; i < 5; ++i)
                {
                    bc[i] = a[j + i];
                }
                for (int i{ 0 }; i < 5; ++i)
                {
                    a[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
                }
            }

            // Iota
            a[0] ^= ROUND_CONSTANTS[round];
        }
    }
}
