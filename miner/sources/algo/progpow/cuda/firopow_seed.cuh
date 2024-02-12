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

    state[10] = 0x00000001;
    state[11] = 0x00;
    state[12] = 0x00;
    state[13] = 0x00;
    state[14] = 0x00;
    state[15] = 0x00;
    state[16] = 0x00;
    state[17] = 0x00;
    state[18] = 0x80008081;
    state[19] = 0x00;
    state[20] = 0x00;
    state[21] = 0x00;
    state[22] = 0x00;
    state[23] = 0x00;
    state[24] = 0x00;

    keccak_f800(state);

    *msb = state[0];
    *lsb = state[1];
}


__device__ __forceinline__
void sha3(
    uint32_t const* const __restrict__ state_init,
    uint4 const* const __restrict__ digest,
    uint32_t* const __restrict__ state_result)
{
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i)
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

    state_result[16] = 0x00;
    state_result[17] = 0x00000001;
    state_result[18] = 0x00;
    state_result[19] = 0x00;
    state_result[20] = 0x00;
    state_result[21] = 0x00;
    state_result[22] = 0x00;
    state_result[23] = 0x00;
    state_result[24] = 0x80008081;

    keccak_f800(state_result);
}