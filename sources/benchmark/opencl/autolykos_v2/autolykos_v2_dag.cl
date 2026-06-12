// =============================================================================
// Autolykos v2 - AMD DAG-fill kernel for the throughput benchmark (self-contained
// snapshot of blake2b_compress + autolykos_v2_dag). Untimed setup that populates
// the ~4 GiB table the search kernel chases. The driver prepends the shared
// common/rotate_byte.cl (rol/ror/bswap helpers) used by the blake2b mixing.
// =============================================================================


__constant
uint BLAKE_2B_SIGMA[12][16] =
{
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 },
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
};


#define G(m, r, i, a, b, c, d)                                                 \
{                                                                              \
    a += b + ((ulong *)m)[BLAKE_2B_SIGMA[r][i]];                               \
    d = ror_u64(d ^ a, 32);                                                    \
    c += d;                                                                    \
    b = ror_u64(b ^ c, 24);                                                    \
    a += b + ((ulong *)m)[BLAKE_2B_SIGMA[r][i + 1]];                           \
    d = ror_u64(d ^ a, 16);                                                    \
    c += d;                                                                    \
    b =  ror_u64(b ^ c, 63);                                                   \
}


#define BLAKE2B_RND(v, r, m)                                                   \
{                                                                              \
    G(m, r, 0, v[ 0],  v[ 4], v[ 8], v[12]);                                   \
    G(m, r, 2, v[ 1],  v[ 5], v[ 9], v[13]);                                   \
    G(m, r, 4, v[ 2],  v[ 6], v[10], v[14]);                                   \
    G(m, r, 6, v[ 3],  v[ 7], v[11], v[15]);                                   \
    G(m, r, 8, v[ 0],  v[ 5], v[10], v[15]);                                   \
    G(m, r, 10, v[ 1], v[ 6], v[11], v[12]);                                   \
    G(m, r, 12, v[ 2], v[ 7], v[ 8], v[13]);                                   \
    G(m, r, 14, v[ 3], v[ 4], v[ 9], v[14]);                                   \
}


inline
void blake2b_compress(ulong *h, const ulong *m, ulong t, ulong f)
{
    ulong v[16];

    ((ulong8 *)v)[0] = ((ulong8 *)h)[0];
    ((ulong8 *)v)[1] = (ulong8)
    (
        0x6A09E667F3BCC908UL, 0xBB67AE8584CAA73BUL,
        0x3C6EF372FE94F82BUL, 0xA54FF53A5F1D36F1UL,
        0x510E527FADE682D1UL, 0x9B05688C2B3E6C1FUL,
        0x1F83D9ABFB41BD6BUL, 0x5BE0CD19137E2179UL
    );

    v[12] ^= t;
    v[14] ^= f;

    #pragma unroll
    for (int rnd = 0; rnd < 12; ++rnd)
    {
        BLAKE2B_RND(v, rnd, m);
    }

    h[0] ^= v[0] ^ v[0 + 8];
    h[1] ^= v[1] ^ v[1 + 8];
    h[2] ^= v[2] ^ v[2 + 8];
    h[3] ^= v[3] ^ v[3 + 8];
    h[4] ^= v[4] ^ v[4 + 8];
    h[5] ^= v[5] ^ v[5 + 8];
    h[6] ^= v[6] ^ v[6 + 8];
    h[7] ^= v[7] ^ v[7 + 8];
}
__kernel
void autolykos_v2_build_dag(
    __global uint* restrict dag,
    uint const              height,
    uint const              period)
{
    uint tid = get_global_id(0);

    if (tid >= period)
        return;

    ulong h[8];
    ulong b[16];
    ulong t = 0;

    //====================================================================//
    //  Initialize context
    //====================================================================//
    ((ulong8 *)h)[0] = (ulong8)
    (
        0x6A09E667F3BCC908UL, 0xBB67AE8584CAA73BUL,
        0x3C6EF372FE94F82BUL, 0xA54FF53A5F1D36F1UL, 
        0x510E527FADE682D1UL, 0x9B05688C2B3E6C1FUL,
        0x1F83D9ABFB41BD6BUL, 0x5BE0CD19137E2179UL
    );
    h[0] ^= 0x01010020;

    //====================================================================//
    //  Hash tid
    //====================================================================//
    ((uint *)b)[0] = as_uint(as_uchar4(tid).s3210);

    //====================================================================//
    //  Hash height
    //====================================================================//
    ((uint *)b)[1] = height;

    //====================================================================//
    //  Hash constant message
    //====================================================================//
    ulong ctr = 0;
    for (int x = 1; x < 16; ++x, ++ctr)
    {
        ((ulong *)b)[x] = as_ulong(as_uchar8(ctr).s76543210);
    }

    #pragma unroll 1
    for (int z = 0; z < 63; ++z)
    {
        t += 128;
        blake2b_compress((ulong *)h, (ulong *)b, t, 0UL);

        #pragma unroll
        for (int x = 0; x < 16; ++x, ++ctr)
        {
            ((ulong *)b)[x] = as_ulong(as_uchar8(ctr).s76543210);
        }
    }

    t += 128;
    blake2b_compress((ulong *)h, (ulong *)b, t, 0UL);

    ((ulong *)b)[0] = as_ulong(as_uchar8(ctr).s76543210);
    t += 8;

    #pragma unroll
    for (int i = 1; i < 16; ++i)
    {
        ((ulong *)b)[i] = 0UL;
    }

    blake2b_compress((ulong *)h, (ulong *)b, t, 0xFFFFFFFFFFFFFFFFUL);

    //====================================================================//
    //  Dump result to global memory -- BIG ENDIAN
    //====================================================================//
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        ((__global ulong *)dag)[(tid + 1) * 4 - i - 1] = as_ulong(as_uchar8(h[i]).s76543210);
    }
    // (ulong) cast is required: tid*32 overflows 32-bit for tid >= 2^27 (the 4 GiB
    // byte boundary), which silently retargets this zeroing write to a low element
    // and leaves the high elements' top byte = H[0] instead of 0 -- corrupting the
    // verify kernel's element sum. See ResolverAutolykosv2AmdTest.dagFullyGeneratedAtLiveHeight1803848.
    // The CUDA path (cuda/dag.cuh) has the same overflow: luminousmining/miner#159.
    ((__global uchar *)dag)[(ulong)tid * 32 + 31] = 0;
}
