#include <common/cast.hpp>
#include <algo/hash.hpp>
#include <algo/rol.hpp>
#include <algo/keccak.hpp>
#include <algo/bitwise.hpp>
#include <algo/ethash/ethash.hpp>


constexpr uint32_t F800_NUMBER_ROUND{ 22u };


static inline void keccak_f1600(
    uint64_t* const state)
{
    constexpr uint64_t roundConstants[24]
    {
        0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
        0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
        0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
        0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
        0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
        0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008
    };

    uint64_t Aba, Abe, Abi, Abo, Abu;
    uint64_t Aga, Age, Agi, Ago, Agu;
    uint64_t Aka, Ake, Aki, Ako, Aku;
    uint64_t Ama, Ame, Ami, Amo, Amu;
    uint64_t Asa, Ase, Asi, Aso, Asu;

    uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
    uint64_t Ega, Ege, Egi, Ego, Egu;
    uint64_t Eka, Eke, Eki, Eko, Eku;
    uint64_t Ema, Eme, Emi, Emo, Emu;
    uint64_t Esa, Ese, Esi, Eso, Esu;

    uint64_t Ba, Be, Bi, Bo, Bu;

    uint64_t Da, De, Di, Do, Du;

    Aba = state[0];
    Abe = state[1];
    Abi = state[2];
    Abo = state[3];
    Abu = state[4];
    Aga = state[5];
    Age = state[6];
    Agi = state[7];
    Ago = state[8];
    Agu = state[9];
    Aka = state[10];
    Ake = state[11];
    Aki = state[12];
    Ako = state[13];
    Aku = state[14];
    Ama = state[15];
    Ame = state[16];
    Ami = state[17];
    Amo = state[18];
    Amu = state[19];
    Asa = state[20];
    Ase = state[21];
    Asi = state[22];
    Aso = state[23];
    Asu = state[24];

    for (size_t n = 0; n < 24; n += 2)
    {
        // Round (n + 0): Axx -> Exx

        Ba = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
        Be = Abe ^ Age ^ Ake ^ Ame ^ Ase;
        Bi = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
        Bo = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
        Bu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;

        Da = Bu ^ algo::rol_u64(Be, 1);
        De = Ba ^ algo::rol_u64(Bi, 1);
        Di = Be ^ algo::rol_u64(Bo, 1);
        Do = Bi ^ algo::rol_u64(Bu, 1);
        Du = Bo ^ algo::rol_u64(Ba, 1);

        Ba = Aba ^ Da;
        Be = algo::rol_u64(Age ^ De, 44);
        Bi = algo::rol_u64(Aki ^ Di, 43);
        Bo = algo::rol_u64(Amo ^ Do, 21);
        Bu = algo::rol_u64(Asu ^ Du, 14);
        Eba = Ba ^ (~Be & Bi) ^ roundConstants[n];
        Ebe = Be ^ (~Bi & Bo);
        Ebi = Bi ^ (~Bo & Bu);
        Ebo = Bo ^ (~Bu & Ba);
        Ebu = Bu ^ (~Ba & Be);

        Ba = algo::rol_u64(Abo ^ Do, 28);
        Be = algo::rol_u64(Agu ^ Du, 20);
        Bi = algo::rol_u64(Aka ^ Da, 3);
        Bo = algo::rol_u64(Ame ^ De, 45);
        Bu = algo::rol_u64(Asi ^ Di, 61);
        Ega = Ba ^ (~Be & Bi);
        Ege = Be ^ (~Bi & Bo);
        Egi = Bi ^ (~Bo & Bu);
        Ego = Bo ^ (~Bu & Ba);
        Egu = Bu ^ (~Ba & Be);

        Ba = algo::rol_u64(Abe ^ De, 1);
        Be = algo::rol_u64(Agi ^ Di, 6);
        Bi = algo::rol_u64(Ako ^ Do, 25);
        Bo = algo::rol_u64(Amu ^ Du, 8);
        Bu = algo::rol_u64(Asa ^ Da, 18);
        Eka = Ba ^ (~Be & Bi);
        Eke = Be ^ (~Bi & Bo);
        Eki = Bi ^ (~Bo & Bu);
        Eko = Bo ^ (~Bu & Ba);
        Eku = Bu ^ (~Ba & Be);

        Ba = algo::rol_u64(Abu ^ Du, 27);
        Be = algo::rol_u64(Aga ^ Da, 36);
        Bi = algo::rol_u64(Ake ^ De, 10);
        Bo = algo::rol_u64(Ami ^ Di, 15);
        Bu = algo::rol_u64(Aso ^ Do, 56);
        Ema = Ba ^ (~Be & Bi);
        Eme = Be ^ (~Bi & Bo);
        Emi = Bi ^ (~Bo & Bu);
        Emo = Bo ^ (~Bu & Ba);
        Emu = Bu ^ (~Ba & Be);

        Ba = algo::rol_u64(Abi ^ Di, 62);
        Be = algo::rol_u64(Ago ^ Do, 55);
        Bi = algo::rol_u64(Aku ^ Du, 39);
        Bo = algo::rol_u64(Ama ^ Da, 41);
        Bu = algo::rol_u64(Ase ^ De, 2);
        Esa = Ba ^ (~Be & Bi);
        Ese = Be ^ (~Bi & Bo);
        Esi = Bi ^ (~Bo & Bu);
        Eso = Bo ^ (~Bu & Ba);
        Esu = Bu ^ (~Ba & Be);


        // Round (n + 1): Exx -> Axx

        Ba = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
        Be = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
        Bi = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
        Bo = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
        Bu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;

        Da = Bu ^ algo::rol_u64(Be, 1);
        De = Ba ^ algo::rol_u64(Bi, 1);
        Di = Be ^ algo::rol_u64(Bo, 1);
        Do = Bi ^ algo::rol_u64(Bu, 1);
        Du = Bo ^ algo::rol_u64(Ba, 1);

        Ba = Eba ^ Da;
        Be = algo::rol_u64(Ege ^ De, 44);
        Bi = algo::rol_u64(Eki ^ Di, 43);
        Bo = algo::rol_u64(Emo ^ Do, 21);
        Bu = algo::rol_u64(Esu ^ Du, 14);
        Aba = Ba ^ (~Be & Bi) ^ roundConstants[n + 1];
        Abe = Be ^ (~Bi & Bo);
        Abi = Bi ^ (~Bo & Bu);
        Abo = Bo ^ (~Bu & Ba);
        Abu = Bu ^ (~Ba & Be);

        Ba = algo::rol_u64(Ebo ^ Do, 28);
        Be = algo::rol_u64(Egu ^ Du, 20);
        Bi = algo::rol_u64(Eka ^ Da, 3);
        Bo = algo::rol_u64(Eme ^ De, 45);
        Bu = algo::rol_u64(Esi ^ Di, 61);
        Aga = Ba ^ (~Be & Bi);
        Age = Be ^ (~Bi & Bo);
        Agi = Bi ^ (~Bo & Bu);
        Ago = Bo ^ (~Bu & Ba);
        Agu = Bu ^ (~Ba & Be);

        Ba = algo::rol_u64(Ebe ^ De, 1);
        Be = algo::rol_u64(Egi ^ Di, 6);
        Bi = algo::rol_u64(Eko ^ Do, 25);
        Bo = algo::rol_u64(Emu ^ Du, 8);
        Bu = algo::rol_u64(Esa ^ Da, 18);
        Aka = Ba ^ (~Be & Bi);
        Ake = Be ^ (~Bi & Bo);
        Aki = Bi ^ (~Bo & Bu);
        Ako = Bo ^ (~Bu & Ba);
        Aku = Bu ^ (~Ba & Be);

        Ba = algo::rol_u64(Ebu ^ Du, 27);
        Be = algo::rol_u64(Ega ^ Da, 36);
        Bi = algo::rol_u64(Eke ^ De, 10);
        Bo = algo::rol_u64(Emi ^ Di, 15);
        Bu = algo::rol_u64(Eso ^ Do, 56);
        Ama = Ba ^ (~Be & Bi);
        Ame = Be ^ (~Bi & Bo);
        Ami = Bi ^ (~Bo & Bu);
        Amo = Bo ^ (~Bu & Ba);
        Amu = Bu ^ (~Ba & Be);

        Ba = algo::rol_u64(Ebi ^ Di, 62);
        Be = algo::rol_u64(Ego ^ Do, 55);
        Bi = algo::rol_u64(Eku ^ Du, 39);
        Bo = algo::rol_u64(Ema ^ Da, 41);
        Bu = algo::rol_u64(Ese ^ De, 2);
        Asa = Ba ^ (~Be & Bi);
        Ase = Be ^ (~Bi & Bo);
        Asi = Bi ^ (~Bo & Bu);
        Aso = Bo ^ (~Bu & Ba);
        Asu = Bu ^ (~Ba & Be);
    }

    state[0] = Aba;
    state[1] = Abe;
    state[2] = Abi;
    state[3] = Abo;
    state[4] = Abu;
    state[5] = Aga;
    state[6] = Age;
    state[7] = Agi;
    state[8] = Ago;
    state[9] = Agu;
    state[10] = Aka;
    state[11] = Ake;
    state[12] = Aki;
    state[13] = Ako;
    state[14] = Aku;
    state[15] = Ama;
    state[16] = Ame;
    state[17] = Ami;
    state[18] = Amo;
    state[19] = Amu;
    state[20] = Asa;
    state[21] = Ase;
    state[22] = Asi;
    state[23] = Aso;
    state[24] = Asu;
}


void algo::keccak(
    uint64_t* const out,
    uint32_t bits,
    uint8_t const* data,
    uint32_t size)
{
    constexpr size_t wordSize{ sizeof(uint64_t) };
    uint32_t const hashSize{ bits / 8 };
    uint32_t const blockSize{ (1600 - bits * 2) / 8 };

    uint64_t* state_iter{ nullptr };
    uint64_t last_word{ 0 };
    uint8_t* last_word_iter{ (uint8_t*)(&last_word) };

    uint64_t state[25] = {0};

    while (size >= blockSize)
    {
        for (uint32_t i{ 0u }; i < (blockSize / wordSize); ++i)
        {
            state[i] ^= le::uint64(data);
            data += wordSize;
        }
        keccak_f1600(state);
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

    keccak_f1600(state);

    uint32_t const maxIndex{ hashSize / castU32(wordSize) };
    for (uint32_t i{ 0u }; i < maxIndex; ++i)
    {
        out[i] = state[i];
    }
}


static inline void keccak_f800(
    uint32_t* out,
    uint32_t round)
{
    constexpr uint32_t KECCAK_INDEX_MAX{ 24u };
    constexpr uint32_t OUT_INDEX_MAX{ 25u };

    static const uint32_t roundConstant[F800_NUMBER_ROUND]
    {
        0x00000001, 0x00008082, 0x0000808a, 0x80008000,
        0x0000808b, 0x80000001, 0x80008081, 0x00008009,
        0x0000008a, 0x00000088, 0x80008009, 0x8000000a,
        0x8000808b, 0x0000008b, 0x00008089, 0x00008003,
        0x00008002, 0x00000080, 0x0000800a, 0x8000000a,
        0x80008081, 0x00008080
    };

    constexpr uint32_t rotc[KECCAK_INDEX_MAX]
    {
        1u,  3u,  6u,  10u, 15u, 21u, 28u, 36u, 45u, 55u, 2u,  14u,
        27u, 41u, 56u, 8u,  25u, 43u, 62u, 18u, 39u, 61u, 20u, 44u
    };

    constexpr uint32_t piln[KECCAK_INDEX_MAX]
    {
        10u, 7u,  11u, 17u, 18u, 3u, 5u,  16u, 8u,  21u, 24u, 4u,
        15u, 23u, 19u, 13u, 12u, 2u, 20u, 14u, 22u, 9u,  6u,  1u
    };

    uint32_t t;
    uint32_t bc[5u];

    // Theta (θ)
    for (uint32_t i{ 0u }; i < 5u; ++i)
    {
        bc[i] =   out[i]
                ^ out[i + 5u]
                ^ out[i + 10u]
                ^ out[i + 15u]
                ^ out[i + 20u];
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
