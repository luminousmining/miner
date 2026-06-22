#include <algo/bitwise.hpp>
#include <algo/ethash/ethash.hpp>
#include <algo/hash.hpp>
#include <algo/keccak.hpp>
#include <algo/rol.hpp>
#include <common/cast.hpp>


constexpr uint32_t F800_NUMBER_ROUND{ 22u };


void algo::keccak(uint64_t* const out, uint32_t bits, uint8_t const* data, uint32_t size)
{
    constexpr size_t wordSize{ sizeof(uint64_t) };
    uint32_t const   hashSize{ bits / 8 };
    uint32_t const   blockSize{ (1600 - bits * 2) / 8 };

    uint64_t* state_iter{ nullptr };
    uint64_t  last_word{ 0 };
    uint8_t*  last_word_iter{ (uint8_t*)(&last_word) };

    uint64_t state[25] = { 0 };

    while (size >= blockSize)
    {
        for (uint32_t i{ 0u }; i < (blockSize / wordSize); ++i)
        {
            state[i] ^= le::uint64(data);
            data += wordSize;
        }
        keccakF1600(state);
        size -= blockSize;
    }

    state_iter = state;

    while (size >= wordSize)
    {
        *state_iter ^= le::uint64(data);
        ++state_iter;
        data += wordSize;
        size -= wordSize;
    }

    while (size > 0u)
    {
        *last_word_iter = *data;
        ++last_word_iter;
        ++data;
        --size;
    }
    *last_word_iter = 0x01;
    *state_iter ^= last_word;

    state[(blockSize / wordSize) - 1] ^= 0x8000000000000000;

    keccakF1600(state);

    uint32_t const maxIndex{ hashSize / castU32(wordSize) };
    for (uint32_t i{ 0u }; i < maxIndex; ++i)
    {
        out[i] = state[i];
    }
}


static inline void keccak_f800(uint32_t* out, uint32_t round)
{
    constexpr uint32_t KECCAK_INDEX_MAX{ 24u };
    constexpr uint32_t OUT_INDEX_MAX{ 25u };

    static const uint32_t roundConstant[F800_NUMBER_ROUND]{ 0x00000001, 0x00008082, 0x0000808a, 0x80008000, 0x0000808b,
                                                            0x80000001, 0x80008081, 0x00008009, 0x0000008a, 0x00000088,
                                                            0x80008009, 0x8000000a, 0x8000808b, 0x0000008b, 0x00008089,
                                                            0x00008003, 0x00008002, 0x00000080, 0x0000800a, 0x8000000a,
                                                            0x80008081, 0x00008080 };

    constexpr uint32_t rotc[KECCAK_INDEX_MAX]{ 1u,  3u,  6u,  10u, 15u, 21u, 28u, 36u, 45u, 55u, 2u,  14u,
                                               27u, 41u, 56u, 8u,  25u, 43u, 62u, 18u, 39u, 61u, 20u, 44u };

    constexpr uint32_t piln[KECCAK_INDEX_MAX]{ 10u, 7u,  11u, 17u, 18u, 3u, 5u,  16u, 8u,  21u, 24u, 4u,
                                               15u, 23u, 19u, 13u, 12u, 2u, 20u, 14u, 22u, 9u,  6u,  1u };

    uint32_t t;
    uint32_t bc[5u];

    // Theta (θ)
    for (uint32_t i{ 0u }; i < 5u; ++i)
    {
        bc[i] = out[i] ^ out[i + 5u] ^ out[i + 10u] ^ out[i + 15u] ^ out[i + 20u];
    }
    for (uint32_t i{ 0u }; i < 5u; i++)
    {
        t = bc[(i + 4u) % 5u] ^ algo::rol_u32(bc[(i + 1u) % 5u], 1u);
        for (uint32_t j{ 0u }; j < OUT_INDEX_MAX; j += 5u)
        {
            out[j + i] ^= t;
        }
    }

    // Rho Pi (ρ, π)
    t = out[1u];
    for (uint32_t i{ 0u }; i < KECCAK_INDEX_MAX; ++i)
    {
        uint32_t j{ piln[i] };
        bc[0u] = out[j];
        out[j] = algo::rol_u32(t, rotc[i]);
        t = bc[0u];
    }

    // Chi (χ)
    for (uint32_t j{ 0u }; j < OUT_INDEX_MAX; j += 5)
    {
        for (uint32_t i{ 0u }; i < 5u; ++i)
        {
            bc[i] = out[j + i];
        }
        for (uint32_t i{ 0u }; i < 5u; ++i)
        {
            out[j + i] ^= (~bc[(i + 1u) % 5u]) & bc[(i + 2u) % 5u];
        }
    }

    //  Iota (ι)
    out[0u] ^= roundConstant[round];
}


algo::hash256 algo::keccak(uint32_t* const out)
{
    for (uint32_t r{ 0u }; r < F800_NUMBER_ROUND; ++r)
    {
        keccak_f800(out, r);
    }

    algo::hash256 hash{};
    for (uint32_t i{ 0 }; i < algo::LEN_HASH_256_WORD_32; ++i)
    {
        hash.word32[i] = out[i];
    }

    return hash;
}


algo::hash256 algo::keccak(algo::hash800& src)
{
    return keccak(src.word32);
}
