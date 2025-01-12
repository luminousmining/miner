__device__ __forceinline__
void create_seed(
    uint4 const* const __restrict__ header,
    uint64_t const nonce,
    uint32_t* const __restrict__ msb,
    uint32_t* const __restrict__ lsb)
{
    uint32_t state[STATE_LEN];

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

    *msb = state[0];
    *lsb = state[1];
}


__device__ __forceinline__
uint64_t sha3(
    uint4 const* const __restrict__ header,
    uint32_t const* const __restrict__ digest,
    uint64_t const seed)
{
    uint32_t state[STATE_LEN];

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

    state[10] = digest[0];
    state[11] = digest[1];
    state[12] = digest[2];
    state[13] = digest[3];
    state[14] = digest[4];
    state[15] = digest[5];
    state[16] = digest[6];
    state[17] = digest[7];

    state[18] = 0u;
    state[19] = 0u;
    state[20] = 0u;
    state[21] = 0u;
    state[22] = 0u;
    state[23] = 0u;
    state[24] = 0u;

    keccak_f800(state);

    return ((uint64_t)(be_u32(state[0]))) << 32 | be_u32(state[1]);
}


__device__ __forceinline__
uint64_t is_valid(
    uint4 const* const __restrict__ header,
    uint32_t * __restrict__ const digest,
    uint64_t const seed)
{

    digest[0] = fnv1a(fnv1a(FNV1_OFFSET, digest[0]), digest[8]);
    digest[1] = fnv1a(fnv1a(FNV1_OFFSET, digest[1]), digest[9]);
    digest[2] = fnv1a(fnv1a(FNV1_OFFSET, digest[2]), digest[10]);
    digest[3] = fnv1a(fnv1a(FNV1_OFFSET, digest[3]), digest[11]);
    digest[4] = fnv1a(fnv1a(FNV1_OFFSET, digest[4]), digest[12]);
    digest[5] = fnv1a(fnv1a(FNV1_OFFSET, digest[5]), digest[13]);
    digest[6] = fnv1a(fnv1a(FNV1_OFFSET, digest[6]), digest[14]);
    digest[7] = fnv1a(fnv1a(FNV1_OFFSET, digest[7]), digest[15]);

    return sha3(header, digest, seed);
}
