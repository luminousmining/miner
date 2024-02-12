inline
void keccak_f1600_round(
    ulong* const state,
    ulong const constantIota)
{
    ulong value;
    ulong C[5];
    ulong D[5];
    ulong tmp[25];

    // THETA
    C[0] = xor5(state, 0);
    C[1] = xor5(state, 1);
    C[2] = xor5(state, 2);
    C[3] = xor5(state, 3);
    C[4] = xor5(state, 4);

    D[0] = rol_u64(C[0], 1u);
    D[1] = rol_u64(C[1], 1u);
    D[2] = rol_u64(C[2], 1u);
    D[3] = rol_u64(C[3], 1u);
    D[4] = rol_u64(C[4], 1u);

    value = D[1] ^ C[4];
    state[0] ^= value;
    state[5] ^= value;
    state[10] ^= value;
    state[15] ^= value;
    state[20] ^= value;

    value = D[2] ^ C[0];
    state[1] ^= value;
    state[6] ^= value;
    state[11] ^= value;
    state[16] ^= value;
    state[21] ^= value;

    value = D[3] ^ C[1];
    state[2] ^= value;
    state[7] ^= value;
    state[12] ^= value;
    state[17] ^= value;
    state[22] ^= value;

    value = D[4] ^ C[2];
    state[3] ^= value;
    state[8] ^= value;
    state[13] ^= value;
    state[18] ^= value;
    state[23] ^= value;

    value = D[0] ^ C[3];
    state[4] ^= value;
    state[9] ^= value;
    state[14] ^= value;
    state[19] ^= value;
    state[24] ^= value;

    // Rho
    tmp[1] = rol_u64(state[1], 1u);
    tmp[2] = rol_u64(state[2], 62u);
    tmp[3] = rol_u64(state[3], 28u);
    tmp[4] = rol_u64(state[4], 27u);
    tmp[5] = rol_u64(state[5], 36u);
    tmp[6] = rol_u64(state[6], 44u);
    tmp[7] = rol_u64(state[7], 6u);
    tmp[8] = rol_u64(state[8], 55u);
    tmp[9] = rol_u64(state[9], 20u);
    tmp[10] = rol_u64(state[10], 3u);
    tmp[11] = rol_u64(state[11], 10u);
    tmp[12] = rol_u64(state[12], 43u);
    tmp[13] = rol_u64(state[13], 25u);
    tmp[14] = rol_u64(state[14], 39u);
    tmp[15] = rol_u64(state[15], 41u);
    tmp[16] = rol_u64(state[16], 45u);
    tmp[17] = rol_u64(state[17], 15u);
    tmp[18] = rol_u64(state[18], 21u);
    tmp[19] = rol_u64(state[19], 8u);
    tmp[20] = rol_u64(state[20], 18u);
    tmp[21] = rol_u64(state[21], 2u);
    tmp[22] = rol_u64(state[22], 61u);
    tmp[23] = rol_u64(state[23], 56u);
    tmp[24] = rol_u64(state[24], 14u);

    // PI
    state[0] = tmp[0];
    state[16] = tmp[5];
    state[7] = tmp[10];
    state[23] = tmp[15];
    state[14] = tmp[20];

    state[10] = tmp[1];
    state[1] = tmp[6];
    state[17] = tmp[11];
    state[8] = tmp[16];
    state[24] = tmp[21];

    state[20] = tmp[2];
    state[11] = tmp[7];
    state[2] = tmp[12];
    state[18] = tmp[17];
    state[9] = tmp[22];

    state[5] = tmp[3];
    state[21] = tmp[8];
    state[12] = tmp[13];
    state[3] = tmp[18];
    state[19] = tmp[23];

    state[15] = tmp[4];
    state[6] = tmp[9];
    state[22] = tmp[14];
    state[13] = tmp[19];
    state[4] = tmp[24];

    // CHI
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 5u; ++i)
    {
        uint const j = i * 5u;
        C[0] = state[j]     ^ ((~state[j + 1]) & state[j + 2]);
        C[1] = state[j + 1] ^ ((~state[j + 2]) & state[j + 3]);
        C[2] = state[j + 2] ^ ((~state[j + 3]) & state[j + 4]);
        C[3] = state[j + 3] ^ ((~state[j + 4]) & state[j]);
        C[4] = state[j + 4] ^ ((~state[j])     & state[j + 1]);

        state[j] = C[0];
        state[j + 1] = C[1];
        state[j + 2] = C[2];
        state[j + 3] = C[3];
        state[j + 4] = C[4];
    }

    // IOTA
    state[0] ^= constantIota;
}


inline
void keccak_f1600(
    uint4* const restrict mix)
{
    ulong state[25];

    state[0] = (((ulong)mix[0].y) << 32) | mix[0].x;
    state[1] = (((ulong)mix[0].w) << 32) | mix[0].z;

    state[2] = (((ulong)mix[1].y) << 32) | mix[1].x;
    state[3] = (((ulong)mix[1].w) << 32) | mix[1].z;

    state[4] = (((ulong)mix[2].y) << 32) | mix[2].x;
    state[5] = (((ulong)mix[2].w) << 32) | mix[2].z;

    state[6] = (((ulong)mix[3].y) << 32) | mix[3].x;
    state[7] = (((ulong)mix[3].w) << 32) | mix[3].z;

    state[8] = 0x8000000000000001UL;

    __attribute__((opencl_unroll_hint))
    for (uint i = 9u; i < 25u; ++i)
    {
        state[i] = 0ul;
    }

    keccak_f1600_round(state, 0x0000000000000001UL);
    keccak_f1600_round(state, 0x0000000000008082UL);
    keccak_f1600_round(state, 0x800000000000808AUL);
    keccak_f1600_round(state, 0x8000000080008000UL);
    keccak_f1600_round(state, 0x000000000000808BUL);
    keccak_f1600_round(state, 0x0000000080000001UL);
    keccak_f1600_round(state, 0x8000000080008081UL);
    keccak_f1600_round(state, 0x8000000000008009UL);
    keccak_f1600_round(state, 0x000000000000008AUL);
    keccak_f1600_round(state, 0x0000000000000088UL);
    keccak_f1600_round(state, 0x0000000080008009UL);
    keccak_f1600_round(state, 0x000000008000000AUL);
    keccak_f1600_round(state, 0x000000008000808BUL);
    keccak_f1600_round(state, 0x800000000000008BUL);
    keccak_f1600_round(state, 0x8000000000008089UL);
    keccak_f1600_round(state, 0x8000000000008003UL);
    keccak_f1600_round(state, 0x8000000000008002UL);
    keccak_f1600_round(state, 0x8000000000000080UL);
    keccak_f1600_round(state, 0x000000000000800AUL);
    keccak_f1600_round(state, 0x800000008000000AUL);
    keccak_f1600_round(state, 0x8000000080008081UL);
    keccak_f1600_round(state, 0x8000000000008080UL);
    keccak_f1600_round(state, 0x0000000080000001UL);
    keccak_f1600_round(state, 0x8000000080008008UL);

    mix[0].x = state[0];
    mix[0].y = state[0] >> 32;
    mix[0].z = state[1];
    mix[0].w = state[1] >> 32;

    mix[1].x = state[2];
    mix[1].y = state[2] >> 32;
    mix[1].z = state[3];
    mix[1].w = state[3] >> 32;

    mix[2].x = state[4];
    mix[2].y = state[4] >> 32;
    mix[2].z = state[5];
    mix[2].w = state[5] >> 32;

    mix[3].x = state[6];
    mix[3].y = state[6] >> 32;
    mix[3].z = state[7];
    mix[3].w = state[7] >> 32;
}
