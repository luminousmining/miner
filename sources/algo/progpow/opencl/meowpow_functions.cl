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

    seed[10] = 'M';
    seed[11] = 'E';
    seed[12] = 'O';
    seed[13] = 'W';

    seed[14] = 'C';
    seed[15] = 'O';
    seed[16] = 'I';
    seed[17] = 'N';

    seed[18] = 'M';
    seed[19] = 'E';
    seed[20] = 'O';
    seed[21] = 'W';
    seed[22] = 'P';
    seed[23] = 'O';
    seed[24] = 'W';

    keccak_f800(seed);
}


inline
ulong sha3(
    uint const* const restrict seed,
    uint* const restrict digest)
{
    uint state[25];

    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 8u; ++i)
    {
        state[i] = seed[i];
    }

    state[8] = digest[0];
    state[9] = digest[1];
    state[10] = digest[2];
    state[11] = digest[3];
    state[12] = digest[4];
    state[13] = digest[5];
    state[14] = digest[6];
    state[15] = digest[7];

    state[16] = 'M';
    state[17] = 'E';
    state[18] = 'O';
    state[19] = 'W';

    state[20] = 'C';
    state[21] = 'O';
    state[22] = 'I';
    state[23] = 'N';

    state[24] = 'M';

    keccak_f800(state);

    ulong const res = ((ulong)state[1]) << 32 | state[0];
    return as_ulong(as_uchar8(res).s76543210);
}


inline
ulong is_valid(
    uint const* const restrict seed,
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

    return sha3(seed, digest);
}
