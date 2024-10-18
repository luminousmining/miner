#pragma once

#define REF_Z00 0
#define REF_Z01 1
#define REF_Z02 2
#define REF_Z03 3
#define REF_Z04 4
#define REF_Z05 5
#define REF_Z06 6
#define REF_Z07 7
#define REF_Z08 8
#define REF_Z09 9
#define REF_Z0A 10
#define REF_Z0B 11
#define REF_Z0C 12
#define REF_Z0D 13
#define REF_Z0E 14
#define REF_Z0F 15
#define REF_Z10 2
#define REF_Z11 6
#define REF_Z12 3
#define REF_Z13 10
#define REF_Z14 7
#define REF_Z15 0
#define REF_Z16 4
#define REF_Z17 13
#define REF_Z18 1
#define REF_Z19 11
#define REF_Z1A 12
#define REF_Z1B 5
#define REF_Z1C 9
#define REF_Z1D 14
#define REF_Z1E 15
#define REF_Z1F 8
#define REF_Z20 3
#define REF_Z21 4
#define REF_Z22 10
#define REF_Z23 12
#define REF_Z24 13
#define REF_Z25 2
#define REF_Z26 7
#define REF_Z27 14
#define REF_Z28 6
#define REF_Z29 5
#define REF_Z2A 9
#define REF_Z2B 0
#define REF_Z2C 11
#define REF_Z2D 15
#define REF_Z2E 8
#define REF_Z2F 1
#define REF_Z30 10
#define REF_Z31 7
#define REF_Z32 12
#define REF_Z33 9
#define REF_Z34 14
#define REF_Z35 3
#define REF_Z36 13
#define REF_Z37 15
#define REF_Z38 4
#define REF_Z39 0
#define REF_Z3A 11
#define REF_Z3B 2
#define REF_Z3C 5
#define REF_Z3D 8
#define REF_Z3E 1
#define REF_Z3F 6
#define REF_Z40 12
#define REF_Z41 13
#define REF_Z42 9
#define REF_Z43 11
#define REF_Z44 15
#define REF_Z45 10
#define REF_Z46 14
#define REF_Z47 8
#define REF_Z48 7
#define REF_Z49 2
#define REF_Z4A 5
#define REF_Z4B 3
#define REF_Z4C 0
#define REF_Z4D 1
#define REF_Z4E 6
#define REF_Z4F 4
#define REF_Z50 9
#define REF_Z51 14
#define REF_Z52 11
#define REF_Z53 5
#define REF_Z54 8
#define REF_Z55 12
#define REF_Z56 15
#define REF_Z57 1
#define REF_Z58 13
#define REF_Z59 3
#define REF_Z5A 0
#define REF_Z5B 10
#define REF_Z5C 2
#define REF_Z5D 6
#define REF_Z5E 4
#define REF_Z5F 7
#define REF_Z60 11
#define REF_Z61 15
#define REF_Z62 5
#define REF_Z63 0
#define REF_Z64 1
#define REF_Z65 9
#define REF_Z66 8
#define REF_Z67 6
#define REF_Z68 14
#define REF_Z69 10
#define REF_Z6A 2
#define REF_Z6B 12
#define REF_Z6C 3
#define REF_Z6D 4
#define REF_Z6E 7
#define REF_Z6F 13


#define REF_Mx(r, i) (buffer[REF_Z##r##i])

#define REF_G(a, b, c, d, x, y)                                                \
    {                                                                          \
        state[a] = state[a] + state[b] + x;                                    \
        state[d] = ror_u32(state[d] ^ state[a], 16);                           \
        state[c] = state[c] + state[d];                                        \
        state[b] = ror_u32(state[b] ^ state[c], 12);                           \
        state[a] = state[a] + state[b] + y;                                    \
        state[d] = ror_u32(state[d] ^ state[a], 8);                            \
        state[c] = state[c] + state[d];                                        \
        state[b] = ror_u32(state[b] ^ state[c], 7);                            \
    }

#define ROUND_S(round)                                                         \
    {                                                                          \
        REF_G(0x0, 0x4, 0x8, 0xC, REF_Mx(round, 0), REF_Mx(round, 1));         \
        REF_G(0x1, 0x5, 0x9, 0xD, REF_Mx(round, 2), REF_Mx(round, 3));         \
        REF_G(0x2, 0x6, 0xA, 0xE, REF_Mx(round, 4), REF_Mx(round, 5));         \
        REF_G(0x3, 0x7, 0xB, 0xF, REF_Mx(round, 6), REF_Mx(round, 7));         \
        REF_G(0x0, 0x5, 0xA, 0xF, REF_Mx(round, 8), REF_Mx(round, 9));         \
        REF_G(0x1, 0x6, 0xB, 0xC, REF_Mx(round, A), REF_Mx(round, B));         \
        REF_G(0x2, 0x7, 0x8, 0xD, REF_Mx(round, C), REF_Mx(round, D));         \
        REF_G(0x3, 0x4, 0x9, 0xE, REF_Mx(round, E), REF_Mx(round, F));         \
    }


__forceinline__ __device__
void blake3_compress_pre(
    uint32_t* const vector,
    uint32_t* const buffer,
    uint32_t const block_len,
    uint32_t const flags)
{
    uint32_t state[16];

    state[0] = vector[0];
    state[1] = vector[1];
    state[2] = vector[2];
    state[3] = vector[3];
    state[4] = vector[4];
    state[5] = vector[5];
    state[6] = vector[6];
    state[7] = vector[7];
    state[8] = algo::blake3::VECTOR_INDEX_0;
    state[9] = algo::blake3::VECTOR_INDEX_1;
    state[10] = algo::blake3::VECTOR_INDEX_2;
    state[11] = algo::blake3::VECTOR_INDEX_3;
    state[12] = 0u;
    state[13] = 0u;
    state[14] = block_len;
    state[15] = flags;

    //THD_PRINT_BUFFER("setted state", state, 16);

    ROUND_S(0); // THD_PRINT_BUFFER("round[0] state", state, 16);
    ROUND_S(1); // THD_PRINT_BUFFER("round[1] state", state, 16);
    ROUND_S(2); // THD_PRINT_BUFFER("round[2] state", state, 16);
    ROUND_S(3); // THD_PRINT_BUFFER("round[3] state", state, 16);
    ROUND_S(4); // THD_PRINT_BUFFER("round[4] state", state, 16);
    ROUND_S(5); // THD_PRINT_BUFFER("round[5] state", state, 16);
    ROUND_S(6); // THD_PRINT_BUFFER("round[6] state", state, 16);

    vector[0] = state[0] ^ state[8];
    vector[1] = state[1] ^ state[9];
    vector[2] = state[2] ^ state[10];
    vector[3] = state[3] ^ state[11];
    vector[4] = state[4] ^ state[12];
    vector[5] = state[5] ^ state[13];
    vector[6] = state[6] ^ state[14];
    vector[7] = state[7] ^ state[15];
}
