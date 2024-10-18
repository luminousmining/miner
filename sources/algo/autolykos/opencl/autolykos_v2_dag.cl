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
    ((__global uchar *)dag)[tid * 32 + 31] = 0;
}
