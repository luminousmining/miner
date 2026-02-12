inline
ulong initialize_seed(
    __constant uint4 const* const restrict header,
    uint* const restrict state_mix,
    ulong const nonce)
{
    state_mix[0] = header[0].x;
    state_mix[1] = header[0].y;
    state_mix[2] = header[0].z;
    state_mix[3] = header[0].w;

    state_mix[4] = header[1].x;
    state_mix[5] = header[1].y;
    state_mix[6] = header[1].z;
    state_mix[7] = header[1].w;

    state_mix[8] = nonce;
    state_mix[9] = (nonce >> 32);

    state_mix[10] = 'r';
    state_mix[11] = 'A';
    state_mix[12] = 'V';
    state_mix[13] = 'E';
    state_mix[14] = 'N';

    state_mix[15] = 'C';
    state_mix[16] = 'O';
    state_mix[17] = 'I';
    state_mix[18] = 'N';

    state_mix[19] = 'K';
    state_mix[20] = 'A';
    state_mix[21] = 'W';
    state_mix[22] = 'P';
    state_mix[23] = 'O';
    state_mix[24] = 'W';

    keccak_f800(state_mix);

    ulong const bytes = ((ulong)state_mix[1]) << 32 | state_mix[0];
    return bytes;
}


inline
ulong sha3(
    uint const* const restrict digest_1,
    uint* const restrict digest_2)
{
    uint state[25];

    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < 8u; ++i)
    {
        state[i] = digest_1[i];
    }

    state[8] = digest_2[0];
    state[9] = digest_2[1];
    state[10] = digest_2[2];
    state[11] = digest_2[3];
    state[12] = digest_2[4];
    state[13] = digest_2[5];
    state[14] = digest_2[6];
    state[15] = digest_2[7];

    state[16] = 'r';
    state[17] = 'A';
    state[18] = 'V';
    state[19] = 'E';
    state[20] = 'N';

    state[21] = 'C';
    state[22] = 'O';
    state[23] = 'I';
    state[24] = 'N';

    keccak_f800(state);

    ulong const res = ((ulong)state[1]) << 32 | state[0];
    return as_ulong(as_uchar8(res).s76543210);
}


inline
ulong is_valid(
    uint const* const restrict state_mix,
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

    return sha3(state_mix, digest);
}


inline
void reduce_hash(
    __local uint* const restrict share_fnv1a,
    uint* const restrict hash,
    uint* const restrict digest,
    uint const worker_group,
    bool const is_same_lane)
{
    uint value = FNV1_OFFSET;

    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < REGS; ++i)
    {
        value = fnv1a_u32(value, hash[i]);
    }
    share_fnv1a[get_global_id(0)] = value;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (true == is_same_lane)
    {
        __attribute__((opencl_unroll_hint))
        for (uint i = 0; i < WORK_ITEM_COLLABORATE; ++i)
        {
            digest[i] = share_fnv1a[worker_group * WORK_ITEM_COLLABORATE + i];
        }
    }
}


inline
void loop_math(
    __global uint4 const* restrict const dag,
    __local uint* restrict const share_hash0,
    __local uint* restrict const header_dag,
    uint* restrict const hash,
    uint const lane_id,
    uint const worker_group)
{
    __attribute__((opencl_unroll_hint(1)))
    for (uint cnt = 0; cnt < COUNT_DAG; ++cnt)
    {
        uint const lane_cnt = cnt % WORK_ITEM_COLLABORATE;
        if (lane_id == lane_cnt)
        {
            share_hash0[worker_group] = hash[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        uint dag_index = share_hash0[worker_group];
        dag_index %= DAG_SIZE;
        dag_index *= WORK_ITEM_COLLABORATE;
        dag_index += ((lane_id ^ cnt) % WORK_ITEM_COLLABORATE);

        uint4 const entries = dag[dag_index];
        sequence_dynamic_local(header_dag, hash, entries);
    }
}


inline
void fill_hash(
    uint* const hash,
    uint const lane_id,
    ulong const seed,
    uint const l_id)
{
    uint4 data;
    data.x = fnv1a_u32(FNV1_OFFSET, (uint)seed);
    data.y = fnv1a_u32(data.x, (uint)(seed >> 32));
    data.z = fnv1a_u32(data.y, lane_id);
    data.w = fnv1a_u32(data.z, lane_id);

    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < REGS; ++i)
    {
        hash[i] = kiss99(&data);
    }
}


inline
void initialize_header(
    __global uint4 const* restrict const dag,
    __local uint * restrict const header_dag,
    uint const thread_id)
{
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < MODULE_LOOP; ++i)
    {
        uint const dag_index = (GROUP_SIZE * i) + thread_id;
        uint const header_index = dag_index * 4u;

        uint4 const item = dag[dag_index];

        // TODO : Bank conflits
        // TODO : Uncoalesced
        header_dag[header_index]      = item.x;
        header_dag[header_index + 1u] = item.y;
        header_dag[header_index + 2u] = item.z;
        header_dag[header_index + 3u] = item.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}


__kernel
void kawpow_lm1(
    __global uint const* const restrict dag,
    __global t_result* const restrict result,
    __constant uint const* const restrict header,
    ulong const start_nonce)
{
    __local uint header_dag[MODULE_CACHE];
    __local ulong share_msb_lsb[SHARE_SEED_SIZE];
    __local uint share_hash0[SHARE_HASH0_SIZE];
    __local uint share_fnv1a[SHARE_FNV1A_SIZE];

    uint state_mix[25];
    uint digest[16];
    uint hash[32];

    uint const thread_id = get_thread_id_2d();
    uint const lane_id = thread_id % WORK_ITEM_COLLABORATE;
    uint const worker_group = get_global_id(0) / WORK_ITEM_COLLABORATE;
    ulong const nonce = start_nonce + thread_id;
    uint const index_share_seed = get_global_id(0) / BATCH_GROUP_LANE;

    ///////////////////////////////////////////////////////////////////////////
    initialize_header(dag, header_dag, (thread_id % GROUP_SIZE));
    ulong const seed = initialize_seed(header, state_mix, nonce);

    __attribute__((opencl_unroll_hint(1)))
    for (uint l_id = 0u; l_id < WORK_ITEM_COLLABORATE; ++l_id)
    {
        ///////////////////////////////////////////////////////////////////////
        if (l_id == lane_id)
        {
            share_msb_lsb[index_share_seed] = seed;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        ////////////////////////////////////////////////////////////////////////
        ulong const seedShare = share_msb_lsb[index_share_seed];

        ////////////////////////////////////////////////////////////////////////
        fill_hash(hash, lane_id, seedShare, l_id);

        loop_math(dag, share_hash0, header_dag, hash, lane_id, worker_group);
        reduce_hash(
            share_fnv1a,
            hash,
            digest,
            worker_group,
            l_id == lane_id);
    }

    ///////////////////////////////////////////////////////////////////////////
    ulong const bytes_result = is_valid(state_mix, digest);
    PRINT_U32_IF("bytes_result", 0u, bytes_result);
    PRINT_U32_IF("bytes_result", 16u, bytes_result);
    if (bytes_result <= 0)
    {
        uint const index = atomic_inc(&result->count);
        if (index < 1)
        {
            result->found = true;
            result->nonce = nonce;
        }
    }
}
