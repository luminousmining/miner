inline
void initialize_seed(
    __constant uint4 const* const restrict header,
    uint* const restrict seed,
    ulong const nonce)
{
    seed[0] = header[0].x;
    seed[1] = header[0].y;
    seed[2] = header[0].z;
    seed[3] = header[0].w;

    seed[4] = header[1].x;
    seed[5] = header[1].y;
    seed[6] = header[1].z;
    seed[7] = header[1].w;

    seed[8] = nonce;
    seed[9] = (nonce >> 32);

    seed[10] = 'r';
    seed[11] = 'A';
    seed[12] = 'V';
    seed[13] = 'E';
    seed[14] = 'N';

    seed[15] = 'C';
    seed[16] = 'O';
    seed[17] = 'I';
    seed[18] = 'N';

    seed[19] = 'K';
    seed[20] = 'A';
    seed[21] = 'W';
    seed[22] = 'P';
    seed[23] = 'O';
    seed[24] = 'W';

    keccak_f800(seed);
}


inline
void sha3(
    uint const* const restrict state_init,
    uint4 const* const restrict digest,
    uint* const restrict state_result)
{
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 8u; ++i)
    {
        state_result[i] = state_init[i];
    }

    state_result[8] = digest[0].x;
    state_result[9] = digest[0].y;
    state_result[10] = digest[0].z;
    state_result[11] = digest[0].w;

    state_result[12] = digest[1].x;
    state_result[13] = digest[1].y;
    state_result[14] = digest[1].z;
    state_result[15] = digest[1].w;

    state_result[16] = 'r';
    state_result[17] = 'A';
    state_result[18] = 'V';
    state_result[19] = 'E';
    state_result[20] = 'N';

    state_result[21] = 'C';
    state_result[22] = 'O';
    state_result[23] = 'I';
    state_result[24] = 'N';

    keccak_f800(state_result);
}
