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
        for (uint i = 0; i < LANES; ++i)
        {
            digest[i] = share_fnv1a[worker_group * LANES + i];
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
        uint const lane_cnt = cnt % LANES;
        if (lane_id == lane_cnt)
        {
            share_hash0[worker_group] = hash[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        uint dag_index = share_hash0[worker_group];
        dag_index %= DAG_SIZE;
        dag_index *= LANES;
        dag_index += ((lane_id ^ cnt) % LANES);

        uint4 const entries = dag[dag_index];
        sequence_dynamic(header_dag, hash, entries);
    }
}


inline
void fill_hash(
    uint* const hash,
    uint const lane_id,
    ulong const seed)
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
void progpow_search(
    ulong const start_nonce,
    ulong const boundary,
    __constant uint4 const* const restrict header,
    __global uint4 const* const restrict dag,
    __global t_result* const restrict result)
{
    __local uint header_dag[MODULE_CACHE];
    __local ulong share_msb_lsb[SHARE_SEED_SIZE];
    __local uint share_hash0[SHARE_HASH0_SIZE];
    __local uint share_fnv1a[SHARE_FNV1A_SIZE];

    uint state_mix[25];
    uint hash[REGS];
    uint digest[LANES];

    uint const thread_id = get_global_id(0) + (get_global_id(1) * GROUP_SIZE);
    uint const lane_id = thread_id % LANES;
    uint const worker_group = get_global_id(0) / LANES;
    ulong nonce = start_nonce + thread_id;
    uint const index_share_seed = get_global_id(0) / BATCH_GROUP_LANE;

    ////////////////////////////////////////////////////////////////////////
    initialize_header(dag, header_dag, (thread_id % GROUP_SIZE));
    ulong const seed = initialize_seed(header, state_mix, nonce);

#if defined(INTERNAL_LOOP)
    __attribute__((opencl_unroll_hint(1)))
    for (uint i = 0; i < INTERNAL_LOOP; ++i)
    {
        nonce += (i * TOTAL_THREADS);
#endif
        __attribute__((opencl_unroll_hint(1)))
        for (uint l_id = 0u; l_id < LANES; ++l_id)
        {
            ////////////////////////////////////////////////////////////////////////
            if (l_id == lane_id)
            {
                share_msb_lsb[index_share_seed] = seed;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            ////////////////////////////////////////////////////////////////////////
            ulong const seedShare = share_msb_lsb[index_share_seed];

            ////////////////////////////////////////////////////////////////////////
            fill_hash(hash, lane_id, seedShare);
            loop_math(dag, share_hash0, header_dag, hash, lane_id, worker_group);
            reduce_hash(
                share_fnv1a,
                hash,
                digest,
                worker_group,
                l_id == lane_id);
        }

        ////////////////////////////////////////////////////////////////////////
    #if defined(__KERNEL_PROGPOW)
        ulong const bytes_result = is_valid(header, digest, seed);
    #else
        ulong const bytes_result = is_valid(state_mix, digest);
    #endif

        if (bytes_result <= boundary)
        {
            uint const index = atomic_inc(&result->count);
            if (index < MAX_RESULT)
            {
                result->found = true;
                result->nonces[index] = nonce;

                result->hash[index][0] = digest[0];
                result->hash[index][1] = digest[1];
                result->hash[index][2] = digest[2];
                result->hash[index][3] = digest[3];
                result->hash[index][4] = digest[4];
                result->hash[index][5] = digest[5];
                result->hash[index][6] = digest[6];
                result->hash[index][7] = digest[7];
            }
        }
#if defined(INTERNAL_LOOP)
    }
#endif
}
