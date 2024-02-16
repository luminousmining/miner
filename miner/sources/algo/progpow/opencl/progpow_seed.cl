inline
void initialize_seed(
    __constant uint4 const* const restrict header,
    uint* const restrict state,
    ulong const nonce)
{
    state[0] = header[0].x;
    state[1] = header[0].y;
    state[2] = header[0].z;
    state[3] = header[0].w;

    state[4] = header[1].x;
    state[5] = header[1].y;
    state[6] = header[1].z;
    state[7] = header[1].w;

    state[8] = nonce;
    state[9] = (nonce >> 32);

    state[10] = 0u;
    state[11] = 0u;
    state[12] = 0u;
    state[13] = 0u;
    state[14] = 0u;
    state[15] = 0u;
    state[16] = 0u;
    state[17] = 0u;
    state[18] = 0u;
    state[19] = 0u;
    state[20] = 0u;
    state[21] = 0u;
    state[22] = 0u;
    state[23] = 0u;
    state[24] = 0u;

    keccak_f800(state);
}


inline
ulong sha3(
    uint4 const* const restrict header,
    uint4 const* const restrict digest,
    uint* const restrict state_result)
{
    uint32_t state[25];

    state[0] = header[0].x;
    state[1] = header[0].y;
    state[2] = header[0].z;
    state[3] = header[0].w;

    state[4] = header[1].x;
    state[5] = header[1].y;
    state[6] = header[1].z;
    state[7] = header[1].w;

    state[8] = nonce;
    state[9] = (nonce >> 32);

    state[10] = header[0].x;
    state[11] = header[0].y;
    state[12] = header[0].z;
    state[13] = header[0].w;
    state[14] = header[1].x;
    state[15] = header[1].y;
    state[16] = header[1].z;
    state[17] = header[1].w;

    state[18] = 0u;
    state[19] = 0u;
    state[20] = 0u;
    state[21] = 0u;
    state[22] = 0u;
    state[23] = 0u;
    state[24] = 0u;

    keccak_f800(state);

    ulong bytes = ((ulong)state_result[1]) << 32 | state_result[0];
    return as_ulong(as_uchar8(bytes).s76543210);
}
