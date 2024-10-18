__kernel
void autolykos_v2_search(
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


    #pragma unroll
    for (int ii = 0; ii < 4; ii++)
    {
        tid = (NONCES_PER_ITER / 4) * ii + get_global_id(0);

        if (tid >= NONCES_PER_ITER)
        {
            break;
        }

        uint CV;
        fn_Add(((uint *)&nonce)[0], tid, 0, non[0], CV);
        non[1] = 0;
        fn_Add(((uint *)&nonce)[1], 0, CV, non[1], CV);

        ulong tmp;
        reverseBytesInt(non[1], ((uint *)(&tmp))[0]);
        reverseBytesInt(non[0], ((uint *)(&tmp))[1]);

        //--------------------------hash
        B2B_IV(aux);
        B2B_IV(aux + 8);
        aux[0] = ivals[0];
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
        hsh = ivals[3];
        hsh ^= ((ulong *)(aux))[3] ^ ((ulong *)(aux))[11];

        reverseBytesInt(((uint*)(&hsh))[1], ((uint *)(&h2))[0]);
        reverseBytesInt(((uint*)(&hsh))[0], ((uint *)(&h2))[1]);

        h3 = h2 % period;

        //--------------------------read hash from lookup
        uint tmpL;
        #pragma unroll 8
        for (int i = 0; i < 32; ++i)
        {
            ((uchar *)r)[31-i] = ((__global uchar const* const)dag)[h3 * 32 + i];
        }

        B2B_IV(aux);
        B2B_IV(aux + 8);
        aux[0] = ivals[0];
        ((ulong *)(aux))[12] ^= 71;//31+32+8;
        ((ulong *)(aux))[13] ^= 0;

        ((ulong *)(aux))[14] = ~((ulong *)(aux))[14];

        uchar bT[72];
        #pragma unroll
        for (j = 0; j < 31; ++j)
        {
            bT[j] = ((uchar *)r)[j + 1];
        }
        #pragma unroll
        for (j = 31; j < 63; ++j)
        {
            bT[j] = ((__global uchar const* const)header)[j - 31];
        }
        #pragma unroll
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

        #pragma unroll
        for (j = 0; j < NUM_SIZE_32; j += 2)
        {
            hsh = ivals[j >> 1];
            hsh ^= ((ulong *)(aux))[j >> 1] ^ ((ulong *)(aux))[8 + (j >> 1)];

            reverseBytesInt(((uint*)(&hsh))[0], r[j]);
            BHashes[THREADS_PER_ITER*j + tid] = r[j];
            reverseBytesInt(((uint*)(&hsh))[1], r[j + 1]);
            BHashes[THREADS_PER_ITER*(j + 1) + tid] = r[j + 1];
        }
    }
}
