///////////////////////////////////////////////////////////////////////////////
#include <benchmark/cuda/common/common.cuh>
#include <benchmark/cuda/kawpow/sequence_math_random.cuh>
#include <benchmark/cuda/kawpow/common.cuh>

///////////////////////////////////////////////////////////////////////////////
#include <benchmark/result.hpp>


__device__ __forceinline__
void create_seed(
    uint64_t nonce,
    uint32_t* const __restrict__ state,
    uint4 const* const __restrict__ header,
    uint32_t* const __restrict__ msb,
    uint32_t* const __restrict__ lsb)
{
    state[0] = header[0].x;
    state[1] = header[0].y;
    state[2] = header[0].z;
    state[3] = header[0].w;

    state[4] = header[1].x;
    state[5] = header[1].y;
    state[6] = header[1].z;
    state[7] = header[1].w;

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

    *msb = state[0];
    *lsb = state[1];
}


__device__ __forceinline__
uint64_t sha3(
    uint32_t const* const __restrict__ state_init,
    uint32_t* __restrict__ const digest)
{
    uint32_t state[STATE_LEN];

    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i)
    {
        state[i] = state_init[i];
    }

    state[8] = digest[0];
    state[9] = digest[1];
    state[10] = digest[2];
    state[11] = digest[3];
    state[12] = digest[4];
    state[13] = digest[5];
    state[14] = digest[6];
    state[15] = digest[7];

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

    return ((uint64_t)(be_u32(state[0]))) << 32 | be_u32(state[1]);
}


__device__ __forceinline__
uint64_t is_valid(
    uint32_t const* __restrict__ const state_init,
    uint32_t* __restrict__ const digest)
{
    digest[0] = fnv1a(fnv1a(FNV1_OFFSET, digest[0]), digest[8]);
    digest[1] = fnv1a(fnv1a(FNV1_OFFSET, digest[1]), digest[9]);
    digest[2] = fnv1a(fnv1a(FNV1_OFFSET, digest[2]), digest[10]);
    digest[3] = fnv1a(fnv1a(FNV1_OFFSET, digest[3]), digest[11]);
    digest[4] = fnv1a(fnv1a(FNV1_OFFSET, digest[4]), digest[12]);
    digest[5] = fnv1a(fnv1a(FNV1_OFFSET, digest[5]), digest[13]);
    digest[6] = fnv1a(fnv1a(FNV1_OFFSET, digest[6]), digest[14]);
    digest[7] = fnv1a(fnv1a(FNV1_OFFSET, digest[7]), digest[15]);

    return sha3(state_init, digest);
}


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
void loop_math(
    uint32_t const lane_id,
    uint4 const* __restrict__ const dag,
    uint32_t* __restrict__ const hash,
    uint32_t const* __restrict__ const header_dag)
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
        sequence_math_random(header_dag, hash, &entries);
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


__device__ __forceinline__
void initialize_header_dag(
    uint32_t* __restrict__ const header_dag,
    uint32_t const* __restrict__ const dag,
    uint32_t const thread_id)
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


__global__
void kernel_kawpow_lm9(
    t_result* const __restrict__ result,
    uint4 const* __restrict__ const header,
    uint4 const* __restrict__ const dag,
    uint64_t const startNonce)
{
    ////////////////////////////////////////////////////////////////////////
    __shared__ uint32_t header_dag[MODULE_CACHE];

    ////////////////////////////////////////////////////////////////////////
    uint32_t hash[REGS];
    uint32_t digest[LANES];
    uint32_t state_init[STATE_LEN];
    uint32_t lsb;
    uint32_t msb;

    ////////////////////////////////////////////////////////////////////////
    uint32_t const thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t const lane_id = threadIdx.x & LANE_ID_MAX;
    uint64_t const nonce = startNonce + thread_id;

    ////////////////////////////////////////////////////////////////////////
    uint32_t const* const dag_u32 = (uint32_t*)dag;
    initialize_header_dag(header_dag, dag_u32, threadIdx.x);

    ////////////////////////////////////////////////////////////////////////
    create_seed(nonce, state_init, header, &lsb, &msb);

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

    uint64_t const bytes = is_valid(state_init, digest);
    if (bytes < 1ull)
    {
        result->found = true;
        uint32_t const index = atomicAdd((uint32_t*)(&result->count), 1);
        if (index < 1)
        {
            result->nonce = nonce;
        }
    }
}


__host__
bool kawpow_lm9(
    cudaStream_t stream,
    t_result* result,
    uint32_t* const header,
    uint32_t* const dag,
    uint32_t const blocks,
    uint32_t const threads)
{
    uint64_t const nonce{ 0ull };

    kernel_kawpow_lm9<<<blocks, threads, 0, stream>>>
    (
        result,
        (uint4*)header,
        (uint4*)dag,
        nonce
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
