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

    seed[10] = 0x00000001;
    seed[11] = 0x00;
    seed[12] = 0x00;
    seed[13] = 0x00;
    seed[14] = 0x00;
    seed[15] = 0x00;
    seed[16] = 0x00;
    seed[17] = 0x00;
    seed[18] = 0x80008081;
    seed[19] = 0x00;
    seed[20] = 0x00;
    seed[21] = 0x00;
    seed[22] = 0x00;
    seed[23] = 0x00;
    seed[24] = 0x00;

    keccak_f800(seed);
}


inline
void sha3(
    uint const* const restrict seed_init,
    uint4 const* const restrict digest,
    uint* const restrict seed_result)
{
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 8u; ++i)
    {
        seed_result[i] = seed_init[i];
    }

    seed_result[8] = digest[0].x;
    seed_result[9] = digest[0].y;
    seed_result[10] = digest[0].z;
    seed_result[11] = digest[0].w;

    seed_result[12] = digest[1].x;
    seed_result[13] = digest[1].y;
    seed_result[14] = digest[1].z;
    seed_result[15] = digest[1].w;

    seed_result[16] = 0x00;
    seed_result[17] = 0x00000001;
    seed_result[18] = 0x00;
    seed_result[19] = 0x00;
    seed_result[20] = 0x00;
    seed_result[21] = 0x00;
    seed_result[22] = 0x00;
    seed_result[23] = 0x00;
    seed_result[24] = 0x80008081;

    keccak_f800(seed_result);
}
