// =============================================================================
// Autolykos v2 - AMD search throughput benchmark kernel (self-contained snapshot).
//
// Frozen copy of the production search assembly with the entry point renamed
// `autolykos_v2_search_lm0`, so the benchmark exercises a snapshot and never the
// live mining kernel. The driver prepends the shared common/rotate_byte.cl
// (rol/ror/bswap helpers); this file supplies the blake2b mixing macros and the
// search body. The NONCES_PER_ITER / THREADS_PER_ITER / NONCE_SIZE_32 /
// NUM_SIZE_32 macros are supplied by the benchmark driver via -D, exactly as
// ResolverAmdAutolykosV2 does.
// =============================================================================


#define B2B_IV(v)                                                              \
{                                                                              \
    ((ulong *)(v))[0] = 0x6A09E667F3BCC908;                                    \
    ((ulong *)(v))[1] = 0xBB67AE8584CAA73B;                                    \
    ((ulong *)(v))[2] = 0x3C6EF372FE94F82B;                                    \
    ((ulong *)(v))[3] = 0xA54FF53A5F1D36F1;                                    \
    ((ulong *)(v))[4] = 0x510E527FADE682D1;                                    \
    ((ulong *)(v))[5] = 0x9B05688C2B3E6C1F;                                    \
    ((ulong *)(v))[6] = 0x1F83D9ABFB41BD6B;                                    \
    ((ulong *)(v))[7] = 0x5BE0CD19137E2179;                                    \
}


// G mixing function
#define B2B_G(v, a, b, c, d, x, y)                                             \
{                                                                              \
    ((ulong *)(v))[a] += ((ulong *)(v))[b] + x;                                \
    ((ulong *)(v))[d] = ror_u64(((ulong *)(v))[d] ^ ((ulong *)(v))[a], 32);    \
    ((ulong *)(v))[c] += ((ulong *)(v))[d];                                    \
    ((ulong *)(v))[b] = ror_u64(((ulong *)(v))[b] ^ ((ulong *)(v))[c], 24);    \
    ((ulong *)(v))[a] += ((ulong *)(v))[b] + y;                                \
    ((ulong *)(v))[d] = ror_u64(((ulong *)(v))[d] ^ ((ulong *)(v))[a], 16);    \
    ((ulong *)(v))[c] += ((ulong *)(v))[d];                                    \
    ((ulong *)(v))[b] = ror_u64(((ulong *)(v))[b] ^ ((ulong *)(v))[c], 63);    \
}


// mixing rounds
#define B2B_MIX(v, m)                                                          \
{                                                                              \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[ 0], ((ulong *)(m))[ 1]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 2], ((ulong *)(m))[ 3]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 4], ((ulong *)(m))[ 5]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[ 6], ((ulong *)(m))[ 7]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 8], ((ulong *)(m))[ 9]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[10], ((ulong *)(m))[11]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[12], ((ulong *)(m))[13]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[14], ((ulong *)(m))[15]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[14], ((ulong *)(m))[10]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 4], ((ulong *)(m))[ 8]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 9], ((ulong *)(m))[15]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[13], ((ulong *)(m))[ 6]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 1], ((ulong *)(m))[12]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[ 0], ((ulong *)(m))[ 2]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[11], ((ulong *)(m))[ 7]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[ 5], ((ulong *)(m))[ 3]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[11], ((ulong *)(m))[ 8]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[12], ((ulong *)(m))[ 0]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 5], ((ulong *)(m))[ 2]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[15], ((ulong *)(m))[13]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[10], ((ulong *)(m))[14]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[ 3], ((ulong *)(m))[ 6]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[ 7], ((ulong *)(m))[ 1]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[ 9], ((ulong *)(m))[ 4]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[ 7], ((ulong *)(m))[ 9]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 3], ((ulong *)(m))[ 1]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[13], ((ulong *)(m))[12]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[11], ((ulong *)(m))[14]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 2], ((ulong *)(m))[ 6]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[ 5], ((ulong *)(m))[10]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[ 4], ((ulong *)(m))[ 0]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[15], ((ulong *)(m))[ 8]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[ 9], ((ulong *)(m))[ 0]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 5], ((ulong *)(m))[ 7]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 2], ((ulong *)(m))[ 4]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[10], ((ulong *)(m))[15]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[14], ((ulong *)(m))[ 1]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[11], ((ulong *)(m))[12]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[ 6], ((ulong *)(m))[ 8]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[ 3], ((ulong *)(m))[13]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[ 2], ((ulong *)(m))[12]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 6], ((ulong *)(m))[10]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 0], ((ulong *)(m))[11]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[ 8], ((ulong *)(m))[ 3]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 4], ((ulong *)(m))[13]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[ 7], ((ulong *)(m))[ 5]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[15], ((ulong *)(m))[14]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[ 1], ((ulong *)(m))[ 9]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[12], ((ulong *)(m))[ 5]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 1], ((ulong *)(m))[15]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[14], ((ulong *)(m))[13]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[ 4], ((ulong *)(m))[10]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 0], ((ulong *)(m))[ 7]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[ 6], ((ulong *)(m))[ 3]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[ 9], ((ulong *)(m))[ 2]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[ 8], ((ulong *)(m))[11]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[13], ((ulong *)(m))[11]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 7], ((ulong *)(m))[14]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[12], ((ulong *)(m))[ 1]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[ 3], ((ulong *)(m))[ 9]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 5], ((ulong *)(m))[ 0]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[15], ((ulong *)(m))[ 4]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[ 8], ((ulong *)(m))[ 6]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[ 2], ((ulong *)(m))[10]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[ 6], ((ulong *)(m))[15]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[14], ((ulong *)(m))[ 9]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[11], ((ulong *)(m))[ 3]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[ 0], ((ulong *)(m))[ 8]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[12], ((ulong *)(m))[ 2]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[13], ((ulong *)(m))[ 7]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[ 1], ((ulong *)(m))[ 4]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[10], ((ulong *)(m))[ 5]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[10], ((ulong *)(m))[ 2]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 8], ((ulong *)(m))[ 4]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 7], ((ulong *)(m))[ 6]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[ 1], ((ulong *)(m))[ 5]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[15], ((ulong *)(m))[11]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[ 9], ((ulong *)(m))[14]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[ 3], ((ulong *)(m))[12]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[13], ((ulong *)(m))[ 0]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[ 0], ((ulong *)(m))[ 1]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 2], ((ulong *)(m))[ 3]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 4], ((ulong *)(m))[ 5]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[ 6], ((ulong *)(m))[ 7]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 8], ((ulong *)(m))[ 9]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[10], ((ulong *)(m))[11]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[12], ((ulong *)(m))[13]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[14], ((ulong *)(m))[15]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[14], ((ulong *)(m))[10]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 4], ((ulong *)(m))[ 8]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 9], ((ulong *)(m))[15]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[13], ((ulong *)(m))[ 6]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 1], ((ulong *)(m))[12]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[ 0], ((ulong *)(m))[ 2]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[11], ((ulong *)(m))[ 7]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[ 5], ((ulong *)(m))[ 3]);            \
}
#define REVERSE_BYTES_INT(input,output)                                          \
{                                                                              \
    void * p = &input;                                                         \
    uchar4 bytesr = ((uchar4 *)p)[0].wzyx;                                     \
    output = *((uint *)&bytesr);                                               \
}


#define FN_ADD(Val1, Val2, cv, Result,ret)                                     \
{                                                                              \
    ulong tmp = (ulong)Val1 + (ulong)Val2 + (ulong)cv;                         \
    Result = tmp;                                                              \
    ret = tmp >> 32;                                                           \
}


__constant
ulong IVALS[8] =
{
    0x6A09E667F2BDC928, 0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
    0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
};


__kernel
void autolykos_v2_search_lm0(
    __global uint const* const restrict header,
    __global uint const* const restrict dag,
    __global uint* const restrict       BHashes,
    ulong const                         nonce,
    uint  const                         period)
{
    uint tid;
    uint r[9] = { 0 };
    ulong aux[32];
    uint j;
    uint non[NONCE_SIZE_32];
    ulong tmp;
    ulong hsh;
    ulong h2;
    uint h3;


    __attribute__((opencl_unroll_hint))
    for (int ii = 0; ii < 4; ii++)
    {
        tid = (NONCES_PER_ITER / 4) * ii + get_global_id(0);

        if (tid >= NONCES_PER_ITER)
        {
            break;
        }

        uint CV;
        FN_ADD(((uint *)&nonce)[0], tid, 0, non[0], CV);
        non[1] = 0;
        FN_ADD(((uint *)&nonce)[1], 0, CV, non[1], CV);

        ulong tmp;
        REVERSE_BYTES_INT(non[1], ((uint *)(&tmp))[0]);
        REVERSE_BYTES_INT(non[0], ((uint *)(&tmp))[1]);

        //--------------------------hash
        B2B_IV(aux);
        B2B_IV(aux + 8);
        aux[0] = IVALS[0];
        ((ulong *)(aux))[12] ^= 40;
        ((ulong *)(aux))[13] ^= 0;

        ((ulong *)(aux))[14] = ~((ulong *)(aux))[14];

        ((ulong *)(aux))[16] = ((__global ulong const* const)header)[0];
        ((ulong *)(aux))[17] = ((__global ulong const* const)header)[1];
        ((ulong *)(aux))[18] = ((__global ulong const* const)header)[2];
        ((ulong *)(aux))[19] = ((__global ulong const* const)header)[3];
        ((ulong *)(aux))[20] = tmp;
        ((ulong *)(aux))[21] = 0;
        ((ulong *)(aux))[22] = 0;
        ((ulong *)(aux))[23] = 0;
        ((ulong *)(aux))[24] = 0;
        ((ulong *)(aux))[25] = 0;
        ((ulong *)(aux))[26] = 0;
        ((ulong *)(aux))[27] = 0;
        ((ulong *)(aux))[28] = 0;
        ((ulong *)(aux))[29] = 0;
        ((ulong *)(aux))[30] = 0;
        ((ulong *)(aux))[31] = 0;

        B2B_MIX(aux, aux + 16);

        ulong hsh;
        hsh = IVALS[3];
        hsh ^= ((ulong *)(aux))[3] ^ ((ulong *)(aux))[11];

        REVERSE_BYTES_INT(((uint*)(&hsh))[1], ((uint *)(&h2))[0]);
        REVERSE_BYTES_INT(((uint*)(&hsh))[0], ((uint *)(&h2))[1]);

        h3 = h2 % period;

        //--------------------------read hash from lookup
        uint tmpL;
        // (ulong) cast is required: h3*32 overflows 32-bit for h3 >= 2^27, which would
        // read the wrong DAG element (and thus a wrong first-hash element f) for any
        // nonce whose seed index lands in the upper part of a >4 GiB table.
        ulong const dagByteBase = (ulong)h3 * 32;
        __attribute__((opencl_unroll_hint(8)))
        for (int i = 0; i < 32; ++i)
        {
            ((uchar *)r)[31-i] = ((__global uchar const* const)dag)[dagByteBase + i];
        }

        B2B_IV(aux);
        B2B_IV(aux + 8);
        aux[0] = IVALS[0];
        ((ulong *)(aux))[12] ^= 71;//31+32+8;
        ((ulong *)(aux))[13] ^= 0;

        ((ulong *)(aux))[14] = ~((ulong *)(aux))[14];

        uchar bT[72];
        __attribute__((opencl_unroll_hint))
        for (j = 0; j < 31; ++j)
        {
            bT[j] = ((uchar *)r)[j + 1];
        }
        __attribute__((opencl_unroll_hint))
        for (j = 31; j < 63; ++j)
        {
            bT[j] = ((__global uchar const* const)header)[j - 31];
        }
        __attribute__((opencl_unroll_hint))
        for (j = 63; j < 71; ++j)
        {
            bT[j] = ((uchar *)&tmp)[j - 63];
        }
        bT[71] = 0;

        ((ulong *)(aux))[16] = ((ulong *)bT)[0];
        ((ulong *)(aux))[17] = ((ulong *)bT)[1];
        ((ulong *)(aux))[18] = ((ulong *)bT)[2];
        ((ulong *)(aux))[19] = ((ulong *)bT)[3];
        ((ulong *)(aux))[20] = ((ulong *)bT)[4];
        ((ulong *)(aux))[21] = ((ulong *)bT)[5];
        ((ulong *)(aux))[22] = ((ulong *)bT)[6];
        ((ulong *)(aux))[23] = ((ulong *)bT)[7];
        ((ulong *)(aux))[24] = ((ulong *)bT)[8];

        ((ulong *)(aux))[25] = 0;
        ((ulong *)(aux))[26] = 0;
        ((ulong *)(aux))[27] = 0;
        ((ulong *)(aux))[28] = 0;
        ((ulong *)(aux))[29] = 0;
        ((ulong *)(aux))[30] = 0;
        ((ulong *)(aux))[31] = 0;

        B2B_MIX(aux, aux + 16);

        __attribute__((opencl_unroll_hint))
        for (j = 0; j < NUM_SIZE_32; j += 2)
        {
            hsh = IVALS[j >> 1];
            hsh ^= ((ulong *)(aux))[j >> 1] ^ ((ulong *)(aux))[8 + (j >> 1)];

            REVERSE_BYTES_INT(((uint*)(&hsh))[0], r[j]);
            BHashes[THREADS_PER_ITER*j + tid] = r[j];
            REVERSE_BYTES_INT(((uint*)(&hsh))[1], r[j + 1]);
            BHashes[THREADS_PER_ITER*(j + 1) + tid] = r[j + 1];
        }
    }
}
