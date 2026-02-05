inline
void initialize_state(
    __constant uint const* restrict header,
    uint* const restrict state,
    ulong const nonce)
{
    __attribute__((opencl_unroll_hint))
    for (uint i = 0; i < 8; ++i)
    {
        state[i] = header[i];
    }

    state[8] = nonce;
    state[9] = (nonce >> 32);

    state[10] = 'r';
    state[11] = 'A';
    state[12] = 'V';
    state[13] = 'E';
    state[14] = 'N';

    state[15] = 'C';
    state[16] = 'O';
    state[17] = 'I';
    state[18] = 'N';

    state[19] = 'K';
    state[20] = 'A';
    state[21] = 'W';
    state[22] = 'P';
    state[23] = 'O';
    state[24] = 'W';

    keccak_f800(state);
}


inline
void fill_hash(
    uint* const restrict hash,
    uint const lane_id,
    uint const lsb,
    uint const msb)
{
    uint4 data;

    data.x = fnv1a_u32(FNV1_OFFSET, lsb);
    data.y = fnv1a_u32(data.x, msb);
    data.z = fnv1a_u32(data.y, lane_id);
    data.w = fnv1a_u32(data.z, lane_id);

    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < REGS; ++i)
    {
        hash[i] = kiss99(&data);
    }
}


inline
void loop_math(
    uint* const restrict dag,
    uint* const restrict hash,
    uint const lane_id)
{
    __attribute__((opencl_unroll_hint(1)))
    for (uint cnt = 0u; cnt < COUNT_DAG; ++cnt)
    {
        uint const mix0 = hash[0];
        uint dag_index = sub_group_broadcast(mix0, cnt % WAVEFRONT);
        dag_index %= DAG_SIZE;
        dag_index *= LANES;
        dag_index += ((lane_id ^ cnt) % LANES);

        uint4 entries;
        entries.x = dag[lane_id];
        entries.y = dag[lane_id];
        entries.z = dag[lane_id];
        entries.w = dag[lane_id];

        sequence_dynamic(dag, hash, entries);
    }
}


inline
void reduce_hash(
    uint* const restrict hash,
    uint* const restrict digest,
    bool const is_same_lane)
{
    uint value = FNV1_OFFSET;

    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < REGS; ++i)
    {
        value = fnv1a_u32(value, hash[i]);
    }

    uint tmp[LANES];
    __attribute__((opencl_unroll_hint))
    for (uint i = 0u; i < LANES; ++i)
    {
        tmp[i] = sub_group_broadcast(value, i);
    }

    if (true == is_same_lane)
    {
        __attribute__((opencl_unroll_hint))
        for (uint i = 0u; i < LANES; ++i)
        {
            digest[i] = tmp[i];
        }
    }
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
    uint const* const restrict state,
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

    return sha3(state, digest);
}


__kernel
void kawpow_lm2(
    ulong const start_nonce,
    __constant uint const* const restrict header,
    __global uint const* const restrict dag,
    __global t_result* const restrict result)
{
    ///////////////////////////////////////////////////////////////////////////
    uint state[25];
    uint hash[REGS];
    uint digest[LANES];

    ///////////////////////////////////////////////////////////////////////////
    uint const thread_id = get_global_id(0) + (get_global_id(1) * GROUP_SIZE);
    uint const lane_id = thread_id % LANES;
    ulong const nonce = start_nonce + thread_id;

    ///////////////////////////////////////////////////////////////////////////
    initialize_state(header, state, nonce);
    uint msb = state[0];
    uint lsb = state[1];

    ///////////////////////////////////////////////////////////////////////////
    __attribute__((opencl_unroll_hint(1)))
    for (uint l_id = 0u; l_id < LANES; ++l_id)
    {
        uint const lane_lsb = sub_group_broadcast(lsb, l_id);
        uint const lane_msb = sub_group_broadcast(msb, l_id);
        fill_hash(hash, lane_id, lane_lsb, lane_msb);
        loop_math(dag, hash, lane_id);
        reduce_hash(hash, digest, l_id == lane_id);
    }

    ///////////////////////////////////////////////////////////////////////////
    ulong const bytes_result = is_valid(state, digest);
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
