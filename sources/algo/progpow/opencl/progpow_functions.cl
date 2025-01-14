inline
ulong initialize_seed(
    __constant uint4 const* const restrict header,
    uint* const restrict state_mix,
    ulong const nonce)
{
    state_mix[0] = header[0].x;
    state_mix[1] = header[0].y;
    state_mix[2] = header[0].z;
    state_mix[3] = header[0].w;

    state_mix[4] = header[1].x;
    state_mix[5] = header[1].y;
    state_mix[6] = header[1].z;
    state_mix[7] = header[1].w;

    state_mix[8] = nonce;
    state_mix[9] = (nonce >> 32);

    state_mix[10] = 0u;
    state_mix[11] = 0u;
    state_mix[12] = 0u;
    state_mix[13] = 0u;
    state_mix[14] = 0u;
    state_mix[15] = 0u;
    state_mix[16] = 0u;
    state_mix[17] = 0u;
    state_mix[18] = 0u;
    state_mix[19] = 0u;
    state_mix[20] = 0u;
    state_mix[21] = 0u;
    state_mix[22] = 0u;
    state_mix[23] = 0u;
    state_mix[24] = 0u;

    keccak_f800(state_mix);

    ulong const bytes = ((ulong)state_mix[1]) << 32 | state_mix[0];
    return as_ulong(as_uchar8(bytes).s76543210);
}


inline
ulong sha3(
    __constant uint4 const* const restrict header,
    uint4 const* const restrict digest,
    ulong const seed)
{
    uint state[25];

    state[0] = header[0].x;
    state[1] = header[0].y;
    state[2] = header[0].z;
    state[3] = header[0].w;

    state[4] = header[1].x;
    state[5] = header[1].y;
    state[6] = header[1].z;
    state[7] = header[1].w;

    state[8] = seed;
    state[9] = (seed >> 32);

    state[10] = digest[0].x;
    state[11] = digest[0].y;
    state[12] = digest[0].z;
    state[13] = digest[0].w;
    state[14] = digest[1].x;
    state[15] = digest[1].y;
    state[16] = digest[1].z;
    state[17] = digest[1].w;

    state[18] = 0u;
    state[19] = 0u;
    state[20] = 0u;
    state[21] = 0u;
    state[22] = 0u;
    state[23] = 0u;
    state[24] = 0u;

    keccak_f800(state);

    ulong bytes = ((ulong)state[1]) << 32 | state[0];
    return as_ulong(as_uchar8(bytes).s76543210);
}


inline
ulong is_valid(
    __constant uint4 const* const restrict header,
    uint* const restrict digest,
    ulong const seed)
{
    digest[0] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[0]), digest[8]);
    digest[1] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[1]), digest[9]);
    digest[2] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[2]), digest[10]);
    digest[3] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[3]), digest[11]);
    digest[4] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[4]), digest[12]);
    digest[5] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[5]), digest[13]);
    digest[6] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[6]), digest[14]);
    digest[7] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[7]), digest[15]);

    return sha3(header, digest, seed);
}
