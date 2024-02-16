#pragma once

#define LANES 16
#define MODULE_CACHE 4096
#define REGS 32


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
    uint32_t const group_id,
    uint32_t* __restrict__ const header_dag,
    uint32_t const* __restrict__ const dag)
{
    #pragma unroll
    for (uint32_t i = 0u; i < 16u; ++i)
    {
        uint32_t const indexDAG = i * 256u + group_id;
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
    uint32_t mix0;

    #pragma unroll 1
    for (uint32_t cnt = 0u; cnt < COUNT_DAG; ++cnt)
    {
        mix0 = hash[0];

        uint32_t dagIndex = __shfl_sync(0xffffffff, mix0, cnt % LANES, LANES);
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
        tmp[i] = __shfl_sync(0xffffffff, value, i, LANES);
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

#if defined(__KERNEL_PROGPOW)
__device__ __forceinline__
void check_result(
    uint32_t const* __restrict__ const state_init,
    uint32_t const* __restrict__ const hash,
    volatile algo::progpow::Result* __restrict__ const result,
    uint64_t const nonce,
    uint64_t const boundary,
    uint64_t const seed)
{
    uint64_t const bytes_result = sha3(state_init, digest, seed);
    if (bytes_result < boundary)
    {
        uint32_t const index = atomicAdd((uint32_t*)(&result->count), 1);
        if (index < 4u)
        {
            result->found = true;
            result->nonces[index] = nonce;

            result->hash[index][0] = digest[0].x;
            result->hash[index][1] = digest[0].y;
            result->hash[index][2] = digest[0].z;
            result->hash[index][3] = digest[0].w;
            result->hash[index][4] = digest[1].x;
            result->hash[index][5] = digest[1].y;
            result->hash[index][6] = digest[1].z;
            result->hash[index][7] = digest[1].w;
        }
    }
}
#elif defined(__KERNEL_KAWPOW) || defined(__KERNEL_FIROPOW)
__device__ __forceinline__
void check_result(
    uint32_t const* __restrict__ const state_init,
    uint32_t const* __restrict__ const hash,
    volatile algo::progpow::Result* __restrict__ const result,
    uint64_t const nonce,
    uint64_t const boundary)
{
    uint4 digest[2];

    digest[0].x = fnv1a(fnv1a(FNV1_OFFSET, hash[0]), hash[8]);
    digest[0].y = fnv1a(fnv1a(FNV1_OFFSET, hash[1]), hash[9]);
    digest[0].z = fnv1a(fnv1a(FNV1_OFFSET, hash[2]), hash[10]);
    digest[0].w = fnv1a(fnv1a(FNV1_OFFSET, hash[3]), hash[11]);

    digest[1].x = fnv1a(fnv1a(FNV1_OFFSET, hash[4]), hash[12]);
    digest[1].y = fnv1a(fnv1a(FNV1_OFFSET, hash[5]), hash[13]);
    digest[1].z = fnv1a(fnv1a(FNV1_OFFSET, hash[6]), hash[14]);
    digest[1].w = fnv1a(fnv1a(FNV1_OFFSET, hash[7]), hash[15]);

    uint32_t state_result[STATE_LEN];
    sha3(state_init, digest, seed);
    uint64_t const bytes_result = ((uint64_t)(be_u32(state_result[0]))) << 32 | be_u32(state_result[1]);

    if (bytes_result < boundary)
    {
        uint32_t const index = atomicAdd((uint32_t*)(&result->count), 1);
        if (index < 4u)
        {
            result->found = true;
            result->nonces[index] = nonce;

            result->hash[index][0] = digest[0].x;
            result->hash[index][1] = digest[0].y;
            result->hash[index][2] = digest[0].z;
            result->hash[index][3] = digest[0].w;
            result->hash[index][4] = digest[1].x;
            result->hash[index][5] = digest[1].y;
            result->hash[index][6] = digest[1].z;
            result->hash[index][7] = digest[1].w;
        }
    }
}
#endif


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
#if defined(__KERNEL_KAWPOW) || defined(__KERNEL_FIROPOW)
    uint32_t state_init[STATE_LEN];
#endif
    uint32_t lsb;
    uint32_t msb;
    uint32_t hash[REGS];
    uint32_t digest[LANES];

    ////////////////////////////////////////////////////////////////////////
    uint32_t const thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t const group_id = get_lane_id();
    uint32_t const lane_id = threadIdx.x & 15; // (LANES - 1)
    uint64_t const nonce = startNonce + thread_id;

    ////////////////////////////////////////////////////////////////////////
    initialize_header_dag(threadIdx.x, header_dag, (uint32_t const* const)dag);
#if defined(__KERNEL_PROGPOW)
    create_seed(header, nonce, &lsb, &msb);
#elif defined(__KERNEL_KAWPOW) || defined(__KERNEL_FIROPOW)
    create_seed(nonce, state_init, header, &lsb, &msb);
#endif

    ////////////////////////////////////////////////////////////////////////
    #pragma unroll 1
    for (uint32_t l_id = 0u; l_id < LANES; ++l_id)
    {
        uint32_t const lane_lsb = __shfl_sync(0xffffffff, lsb, l_id, LANES);
        uint32_t const lane_msb = __shfl_sync(0xffffffff, msb, l_id, LANES);
        fill_hash(lane_id, lane_lsb, lane_msb, hash);
        loop_math(lane_id, dag, hash, header_dag);
        reduce_hash(l_id == lane_id, hash, digest);
    }

    ////////////////////////////////////////////////////////////////////////
#if defined(__KERNEL_PROGPOW)
    uint64_t const seed { ((uint64_t)(be_u32(st[0])))<< 32 | be_u32(st[1]) };
    check_result(header, digest, result, seed, nonce, boundary);
#elif defined(__KERNEL_KAWPOW) || defined(__KERNEL_FIROPOW)
    check_result(state_init, digest, result, nonce, boundary);
#endif
}
