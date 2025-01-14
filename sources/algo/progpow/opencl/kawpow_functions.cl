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

    state_mix[10] = 'r';
    state_mix[11] = 'A';
    state_mix[12] = 'V';
    state_mix[13] = 'E';
    state_mix[14] = 'N';

    state_mix[15] = 'C';
    state_mix[16] = 'O';
    state_mix[17] = 'I';
    state_mix[18] = 'N';

    state_mix[19] = 'K';
    state_mix[20] = 'A';
    state_mix[21] = 'W';
    state_mix[22] = 'P';
    state_mix[23] = 'O';
    state_mix[24] = 'W';

    keccak_f800(state_mix);

    ulong const bytes = ((ulong)state_mix[1]) << 32 | state_mix[0];
    return bytes;
}


inline
ulong sha3(
    uint const* const restrict digest_1,
    uint* const restrict digest_2)
{
    uint state[25];

    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 8u; ++i)
    {
        state[i] = digest_1[i];
    }

    state[8] = digest_2[0];
    state[9] = digest_2[1];
    state[10] = digest_2[2];
    state[11] = digest_2[3];
    state[12] = digest_2[4];
    state[13] = digest_2[5];
    state[14] = digest_2[6];
    state[15] = digest_2[7];

    state[16] = 'r';
    state[17] = 'A';
    state[18] = 'V';
    state[19] = 'E';
    state[20] = 'N';

    state[21] = 'C';
    state[22] = 'O';
    state[23] = 'I';
    state[24] = 'N';

    keccak_f800(state);

    ulong const res = ((ulong)state[1]) << 32 | state[0];
    return as_ulong(as_uchar8(res).s76543210);
}


inline
ulong is_valid(
    uint const* const restrict state_mix,
    uint* const restrict digest)
{
    digest[0] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[0]), digest[8]);
    digest[1] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[1]), digest[9]);
    digest[2] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[2]), digest[10]);
    digest[3] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[3]), digest[11]);
    digest[4] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[4]), digest[12]);
    digest[5] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[5]), digest[13]);
    digest[6] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[6]), digest[14]);
    digest[7] = fnv1a_u32(fnv1a_u32(FNV1_OFFSET, digest[7]), digest[15]);

    return sha3(state_mix, digest);
}
