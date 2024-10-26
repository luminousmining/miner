__device__ __forceinline__
void create_seed(
    uint64_t nonce,
    uint32_t* const __restrict__ state,
    uint4 const* const __restrict__ header,
    uint32_t* const __restrict__ msb,
    uint32_t* const __restrict__ lsb)
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

    state[10] = 'M';
    state[11] = 'E';
    state[12] = 'O';
    state[13] = 'W';

    state[14] = 'C';
    state[15] = 'O';
    state[16] = 'I';
    state[17] = 'N';

    state[18] = 'M';
    state[19] = 'E';
    state[20] = 'O';
    state[21] = 'W';
    state[22] = 'P';
    state[23] = 'O';
    state[24] = 'W';


    keccak_f800(state);

    *msb = state[0];
    *lsb = state[1];
}


__device__ __forceinline__
uint64_t sha3(
    uint32_t const* const __restrict__ state_init,
    uint32_t* __restrict__ const digest)
{
    uint32_t state[STATE_LEN];

    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i)
    {
        state[i] = state_init[i];
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

    return ((uint64_t)(be_u32(state[0]))) << 32 | be_u32(state[1]);
}


__device__ __forceinline__
uint64_t is_valid(
    uint32_t const* __restrict__ const state_init,
    uint32_t* __restrict__ const digest)
{
    digest[0] = fnv1a(fnv1a(FNV1_OFFSET, digest[0]), digest[8]);
    digest[1] = fnv1a(fnv1a(FNV1_OFFSET, digest[1]), digest[9]);
    digest[2] = fnv1a(fnv1a(FNV1_OFFSET, digest[2]), digest[10]);
    digest[3] = fnv1a(fnv1a(FNV1_OFFSET, digest[3]), digest[11]);
    digest[4] = fnv1a(fnv1a(FNV1_OFFSET, digest[4]), digest[12]);
    digest[5] = fnv1a(fnv1a(FNV1_OFFSET, digest[5]), digest[13]);
    digest[6] = fnv1a(fnv1a(FNV1_OFFSET, digest[6]), digest[14]);
    digest[7] = fnv1a(fnv1a(FNV1_OFFSET, digest[7]), digest[15]);

    return sha3(state_init, digest);
}
