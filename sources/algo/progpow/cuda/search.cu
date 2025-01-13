#pragma once


__device__ __forceinline__
void fill_hash(
    uint32_t lane_id,
    uint32_t lsb,
    uint32_t msb,
    uint32_t* __restrict__ const hash)
{
    uint4 data;

    data.x = fnv1a(FNV1_OFFSET, lsb);
    data.y = fnv1a(data.x, msb);
    data.z = fnv1a(data.y, lane_id);
    data.w = fnv1a(data.z, lane_id);

    #pragma unroll
    for (uint32_t i = 0u; i < REGS; ++i)
    {
        hash[i] = kiss99(data);
    }
}


__device__ __forceinline__
void initialize_header_dag(
    uint32_t const thread_id,
    uint32_t* __restrict__ const header_dag,
    uint32_t const* __restrict__ const dag)
{
    #pragma unroll
    for (uint32_t i = 0u; i < HEADER_ITEM_BY_THREAD; ++i)
    {
        uint32_t const indexDAG = i * THREAD_COUNT + thread_id;
        uint32_t const itemDag = dag[indexDAG];
        header_dag[indexDAG] = itemDag;
    }
    __syncthreads();
}


__device__ __forceinline__
void loop_math(
    uint32_t const lane_id,
    uint4 const* __restrict__ const dag,
    uint32_t* __restrict__ const hash,
    uint32_t* __restrict__ const header_dag)
{
    #pragma unroll 1
    for (uint32_t cnt = 0u; cnt < COUNT_DAG; ++cnt)
    {
        uint32_t const mix0 = hash[0];

        uint32_t dagIndex = reg_load(mix0, cnt % LANES, LANES);
        dagIndex %= DAG_SIZE;
        dagIndex *= LANES;
        dagIndex += ((lane_id ^ cnt) % LANES);

        uint4 entries = dag[dagIndex];
        sequence_dynamic(header_dag, hash, &entries);
    }
}


__device__ __forceinline__
void reduce_hash(
    bool const is_same_lane,
    uint32_t* __restrict__ const hash,
    uint32_t* __restrict__ const digest)
{
    uint32_t value = FNV1_OFFSET;
    #pragma unroll
    for (uint32_t i = 0u; i < REGS; ++i)
    {
        value = fnv1a(value, hash[i]);
    }

    uint32_t tmp[LANES];
    #pragma unroll
    for (uint32_t i = 0u; i < LANES; ++i)
    {
        tmp[i] = reg_load(value, i, LANES);
    }

    if (is_same_lane == true)
    {
        #pragma unroll
        for (uint32_t i = 0u; i < LANES; ++i)
        {
            digest[i] = tmp[i];
        }
    }
}

__global__
void progpowSearch(
    uint64_t const startNonce,
    uint64_t const boundary,
    uint4 const* __restrict__ const header,
    uint4 const* __restrict__ const dag,
    volatile algo::progpow::Result* __restrict__ const result)
{
    ////////////////////////////////////////////////////////////////////////
    __shared__ uint32_t header_dag[MODULE_CACHE];

    ////////////////////////////////////////////////////////////////////////
#if !defined(__KERNEL_PROGPOW)
    uint32_t state_init[STATE_LEN];
#endif
    uint32_t lsb;
    uint32_t msb;
    uint32_t hash[REGS];
    uint32_t digest[LANES];

    ////////////////////////////////////////////////////////////////////////
    uint32_t const thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t const lane_id = threadIdx.x & LANE_ID_MAX;
    uint64_t const nonce = startNonce + thread_id;

    ////////////////////////////////////////////////////////////////////////
    uint32_t const* const dag_u32 = (uint32_t const* const)dag;
    initialize_header_dag(threadIdx.x, header_dag, dag_u32);

#if defined(__KERNEL_PROGPOW)
    create_seed(header, nonce, &msb, &lsb);
    msb = be_u32(msb);
    lsb = be_u32(lsb);
    uint64_t const seed = ((uint64_t)msb) << 32 | lsb;
#else
    create_seed(nonce, state_init, header, &lsb, &msb);
#endif

    ////////////////////////////////////////////////////////////////////////
    #pragma unroll 1
    for (uint32_t l_id = 0u; l_id < LANES; ++l_id)
    {
        uint32_t const lane_lsb = reg_load(lsb, l_id, LANES);
        uint32_t const lane_msb = reg_load(msb, l_id, LANES);
        fill_hash(lane_id, lane_lsb, lane_msb, hash);
        loop_math(lane_id, dag, hash, header_dag);
        reduce_hash(l_id == lane_id, hash, digest);
    }

    ////////////////////////////////////////////////////////////////////////
#if defined(__KERNEL_PROGPOW)
    uint64_t const bytes_result = is_valid(header, digest, seed);
#else
    uint64_t const bytes_result = is_valid(state_init, digest);
#endif

    if (bytes_result < boundary)
    {
        uint32_t const index = atomicAdd((uint32_t*)(&result->count), 1);
        if (index < algo::progpow::MAX_RESULT)
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
}
