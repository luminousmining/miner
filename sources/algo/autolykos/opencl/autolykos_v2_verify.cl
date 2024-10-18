__kernel
void autolykos_v2_verify(
    __global uint const *const restrict  bound,
    __global uint const* const restrict  dag,
    __global uint const* const restrict  BHashes,
    __global t_result* const restrict    result,
    ulong const                          nonce,
    uint  const                          period,
    uint  const                          height)
{
    __local uint shared_index[64];
    __local uint shared_data[512];

    uint const tid = get_global_id(0);
    uint const threadIdx = get_local_id(0);
    uint const thread_id = threadIdx & 7;
    uint const thrdblck_id = threadIdx;
    uint const hash_id = threadIdx >> 3;

    ulong aux[32] = { 0 };
    uint ind[32] = { 0 };
    uint r[9] = { 0 };

    uint4 v1 = { 0,0,0,0 };
    uint4 v2 = { 0,0,0,0 };
    uint4 v3 = { 0,0,0,0 };
    uint4 v4 = { 0,0,0,0 };

    uchar j = 0;

    if (tid >= NONCES_PER_ITER)
    {
        return;
    }

    //================================================================//
    //  Generate indices
    //================================================================//
    #pragma unroll
    for (int k = 0; k < 8; k++)
    {
        r[k] = (BHashes[k*THREADS_PER_ITER + tid]);
    }
    ((uchar *)r)[32] = ((uchar *)r)[0];
    ((uchar *)r)[33] = ((uchar *)r)[1];
    ((uchar *)r)[34] = ((uchar *)r)[2];
    ((uchar *)r)[35] = ((uchar *)r)[3];

    #pragma unroll
    for (int k = 0; k < K_LEN; k += 4)
    {
        ind[k] = r[k >> 2] % period;
        ind[k + 1] = ((r[k >> 2] << 8)  | (r[(k >> 2) + 1] >> 24)) % period;
        ind[k + 2] = ((r[k >> 2] << 16) | (r[(k >> 2) + 1] >> 16)) % period;
        ind[k + 3] = ((r[k >> 2] << 24) | (r[(k >> 2) + 1] >> 8))  % period;
    }

    //================================================================//
    //  Calculate result
    //================================================================//
    shared_index[thrdblck_id] = ind[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    shared_data[(hash_id << 3) + thread_id]       = (dag[(shared_index[hash_id] << 3)      + thread_id]);
    shared_data[(hash_id << 3) + thread_id + 64]  = (dag[(shared_index[hash_id + 8] << 3)  + thread_id]);
    shared_data[(hash_id << 3) + thread_id + 128] = (dag[(shared_index[hash_id + 16] << 3) + thread_id]);
    shared_data[(hash_id << 3) + thread_id + 192] = (dag[(shared_index[hash_id + 24] << 3) + thread_id]);
    shared_data[(hash_id << 3) + thread_id + 256] = (dag[(shared_index[hash_id + 32] << 3) + thread_id]);
    shared_data[(hash_id << 3) + thread_id + 320] = (dag[(shared_index[hash_id + 40] << 3) + thread_id]);
    shared_data[(hash_id << 3) + thread_id + 384] = (dag[(shared_index[hash_id + 48] << 3) + thread_id]);
    shared_data[(hash_id << 3) + thread_id + 448] = (dag[(shared_index[hash_id + 56] << 3) + thread_id]);
    barrier(CLK_LOCAL_MEM_FENCE);


    v1.x = shared_data[(thrdblck_id << 3) + 0];
    v1.y = shared_data[(thrdblck_id << 3) + 1];
    v1.z = shared_data[(thrdblck_id << 3) + 2];
    v1.w = shared_data[(thrdblck_id << 3) + 3];
    v3.x = shared_data[(thrdblck_id << 3) + 4];
    v3.y = shared_data[(thrdblck_id << 3) + 5];
    v3.z = shared_data[(thrdblck_id << 3) + 6];
    v3.w = shared_data[(thrdblck_id << 3) + 7];

    shared_index[thrdblck_id] = ind[1];
    barrier(CLK_LOCAL_MEM_FENCE);

    shared_data[(hash_id << 3) + thread_id]       = (dag[(shared_index[hash_id] << 3)      + thread_id]);
    shared_data[(hash_id << 3) + thread_id + 64]  = (dag[(shared_index[hash_id + 8] << 3)  + thread_id]);
    shared_data[(hash_id << 3) + thread_id + 128] = (dag[(shared_index[hash_id + 16] << 3) + thread_id]);
    shared_data[(hash_id << 3) + thread_id + 192] = (dag[(shared_index[hash_id + 24] << 3) + thread_id]);
    shared_data[(hash_id << 3) + thread_id + 256] = (dag[(shared_index[hash_id + 32] << 3) + thread_id]);
    shared_data[(hash_id << 3) + thread_id + 320] = (dag[(shared_index[hash_id + 40] << 3) + thread_id]);
    shared_data[(hash_id << 3) + thread_id + 384] = (dag[(shared_index[hash_id + 48] << 3) + thread_id]);
    shared_data[(hash_id << 3) + thread_id + 448] = (dag[(shared_index[hash_id + 56] << 3) + thread_id]);
    barrier(CLK_LOCAL_MEM_FENCE);

    v2.x = shared_data[(thrdblck_id << 3) + 0];
    v2.y = shared_data[(thrdblck_id << 3) + 1];
    v2.z = shared_data[(thrdblck_id << 3) + 2];
    v2.w = shared_data[(thrdblck_id << 3) + 3];
    v4.x = shared_data[(thrdblck_id << 3) + 4];
    v4.y = shared_data[(thrdblck_id << 3) + 5];
    v4.z = shared_data[(thrdblck_id << 3) + 6];
    v4.w = shared_data[(thrdblck_id << 3) + 7];


    uint CV = 0;
    fn_Add(v1.x, v2.x, 0, r[0], CV);
    fn_Add(v1.y, v2.y, CV, r[1], CV);
    fn_Add(v1.z, v2.z, CV, r[2], CV);
    fn_Add(v1.w, v2.w, CV, r[3], CV);
    fn_Add(v3.x, v4.x, CV, r[4], CV);
    fn_Add(v3.y, v4.y, CV, r[5], CV);
    fn_Add(v3.z, v4.z, CV, r[6], CV);
    fn_Add(v3.w, v4.w, CV, r[7], CV);
    r[8] = 0; fn_Add(r[8], 0, CV, r[8], CV);


    // remaining additions
    #pragma unroll
    for (int k = 2; k < K_LEN; ++k)
    {
        shared_index[thrdblck_id] = ind[k];
        barrier(CLK_LOCAL_MEM_FENCE);

        shared_data[(hash_id << 3) + thread_id]       = (dag[(shared_index[hash_id] << 3)      + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 64]  = (dag[(shared_index[hash_id + 8] << 3)  + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 128] = (dag[(shared_index[hash_id + 16] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 192] = (dag[(shared_index[hash_id + 24] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 256] = (dag[(shared_index[hash_id + 32] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 320] = (dag[(shared_index[hash_id + 40] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 384] = (dag[(shared_index[hash_id + 48] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 448] = (dag[(shared_index[hash_id + 56] << 3) + thread_id]);
        barrier(CLK_LOCAL_MEM_FENCE);

        v1.x = shared_data[(thrdblck_id << 3) + 0];
        v1.y = shared_data[(thrdblck_id << 3) + 1];
        v1.z = shared_data[(thrdblck_id << 3) + 2];
        v1.w = shared_data[(thrdblck_id << 3) + 3];
        v2.x = shared_data[(thrdblck_id << 3) + 4];
        v2.y = shared_data[(thrdblck_id << 3) + 5];
        v2.z = shared_data[(thrdblck_id << 3) + 6];
        v2.w = shared_data[(thrdblck_id << 3) + 7];

        fn_Add(r[0], v1.x, CV, r[0], CV);
        fn_Add(r[1], v1.y, CV, r[1], CV);
        fn_Add(r[2], v1.z, CV, r[2], CV);
        fn_Add(r[3], v1.w, CV, r[3], CV);
        fn_Add(r[4], v2.x, CV, r[4], CV);
        fn_Add(r[5], v2.y, CV, r[5], CV);
        fn_Add(r[6], v2.z, CV, r[6], CV);
        fn_Add(r[7], v2.w, CV, r[7], CV);
        fn_Add(r[8], 0,    CV, r[8], CV);
    }

    //====================================================================//
    //  Initialize context
    //====================================================================//
    B2B_IV(aux);
    B2B_IV(aux + 8);
    aux[0] = ivals[0];
    ((ulong *)(aux))[12] ^= 32;
    ((ulong *)(aux))[13] ^= 0;

    ((ulong *)(aux))[14] = ~((ulong *)(aux))[14];

    uchar *bb = (uchar *)(&(((ulong *)(aux))[16]));
    for (j = 0; j < NUM_SIZE_8; ++j)
    {
        bb[j] = ((const uchar *)r)[NUM_SIZE_8 - j - 1];
    }

    ((ulong *)(aux))[20] = 0;
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
    #pragma unroll
    for (j = 0; j < NUM_SIZE_32; j += 2)
    {
        hsh = ivals[j >> 1];
        hsh ^= ((ulong *)(aux))[j >> 1] ^ ((ulong *)(aux))[8 + (j >> 1)];
        reverseBytesInt(((uint*)&hsh)[0], r[7 - j]);
        reverseBytesInt(((uint*)&hsh)[1], r[7 - j - 1]);

    }
    //================================================================//
    //  Dump result to global memory -- LITTLE ENDIAN
    //================================================================//

    __global ulong const* const bound64 = (__global ulong const* const)bound;
    ulong const* const r64 = (ulong const* const)r;

    ulong const r3 = r64[3];
    ulong const r2 = r64[2];
    ulong const r1 = r64[1];
    ulong const r0 = r64[0];

    ulong const b3 = bound64[3];
    ulong const b2 = bound64[2];
    ulong const b1 = bound64[1];
    ulong const b0 = bound64[0];

    j =    ((r0 < b0) && (r1 == b1))
        || ((r1 < b1) && (r2 == b2))
        || ((r2 < b2) && (r3 == b3))
        || (r3 < b3);

    if (j)
    {
        uint const index = atomic_inc(&result->count);
        if (index < 4)
        {
            result->found = true;
            result->nonces[index] = nonce + tid;
        }
    }
}
