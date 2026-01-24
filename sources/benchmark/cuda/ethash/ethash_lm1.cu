///////////////////////////////////////////////////////////////////////////////
#include <common/cast.hpp>
#include <benchmark/result.hpp>

///////////////////////////////////////////////////////////////////////////////
#include <benchmark/cuda/common/common.cuh>

///////////////////////////////////////////////////////////////////////////////
#include <common/cuda/be_u64.cuh>
#include <common/cuda/register.cuh>

///////////////////////////////////////////////////////////////////////////////
#include <algo/crypto/cuda/fnv1.cuh>

///////////////////////////////////////////////////////////////////////////////
__constant__ uint4 d_header[2];
__constant__ uint32_t d_dag_number_item;

///////////////////////////////////////////////////////////////////////////////
#include <algo/crypto/cuda/keccak_f1600.cuh>


constexpr uint32_t THREADS_PER_HASH{ 8u };


__device__ __forceinline__
void keccak_f1600_first(
    uint64_t* __restrict__ const state,
    uint4* __restrict__ const seed,
    uint64_t const nonce)
{
    toU64(state, 0u, d_header[0]);
    toU64(state, 2u, d_header[1]);

    state[4] = nonce;
    state[5] = 1ull;
    state[6] = 0ull;
    state[7] = 0ull;
    state[8] = 0x8000000000000000ull;

    #pragma unroll
    for (uint32_t i{ 9u }; i < 25u; ++i)
    {
        state[i] = 0ull;
    }

    #pragma unroll
    for (uint32_t i{ 0u }; i < 23u; ++i)
    {
        keccak_f1600_round(state, i);
    }

    // theta
    uint64_t C[5];
    uint64_t D[5];

    // THETA
    C[0] = xor5(state, 0);
    C[1] = xor5(state, 1);
    C[2] = xor5(state, 2);
    C[3] = xor5(state, 3);
    C[4] = xor5(state, 4);

    D[0] = rol_u64(C[0], 1u);
    D[1] = rol_u64(C[1], 1u);
    D[2] = rol_u64(C[2], 1u);
    D[3] = rol_u64(C[3], 1u);
    D[4] = rol_u64(C[4], 1u);

    state[0]  ^= D[1] ^ C[4];
    state[10] ^= D[1] ^ C[4];

    state[6]  ^= D[2] ^ C[0];
    state[16] ^= D[2] ^ C[0];

    state[12] ^= D[3] ^ C[1];
    state[22] ^= D[3] ^ C[1];

    state[3]  ^= D[4] ^ C[2];
    state[18] ^= D[4] ^ C[2];

    state[9]  ^= D[0] ^ C[3];
    state[24] ^= D[0] ^ C[3];

    // rho pi
    state[1] = rol_u64(state[6],  44u);
    state[6] = rol_u64(state[9],  20u);
    state[9] = rol_u64(state[22], 61u);
    state[2] = rol_u64(state[12], 43u);
    state[4] = rol_u64(state[24], 14u);
    state[8] = rol_u64(state[16], 45u);
    state[5] = rol_u64(state[3],  28u);
    state[3] = rol_u64(state[18], 21u);
    state[7] = rol_u64(state[10], 3u);

    // chi
    uint64_t const f{ state[0] };
    uint64_t const s{ state[1] };
    state[0] = state[0] ^ ((~state[1]) & state[2]);
    state[1] = state[1] ^ ((~state[2]) & state[3]);
    state[2] = state[2] ^ ((~state[3]) & state[4]);
    state[3] = state[3] ^ ((~state[4]) & f);
    state[4] = state[4] ^ ((~f) & s);
    state[5] = state[5] ^ ((~state[6]) & state[7]);
    state[6] = state[6] ^ ((~state[7]) & state[8]);
    state[7] = state[7] ^ ((~state[8]) & state[9]);

    // iota
    state[0] ^= KECCAK_F1600_ROUND[23u];

    // Convert to uint4
    seed[0] = toU4(state[0], state[1]);
    seed[1] = toU4(state[2], state[3]);
    seed[2] = toU4(state[4], state[5]);
    seed[3] = toU4(state[6], state[7]);
}


__device__ __forceinline__
void keccak_f1600_final(
    uint64_t* const state)
{
    state[12] = 1ull;
    state[13] = 0ull;
    state[14] = 0ull;
    state[15] = 0ull;
    state[16] = 0x8000000000000000ull;

    #pragma unroll
    for (uint32_t i{ 17u }; i < 25u; ++i)
    {
        state[i] = 0ull;
    }

    #pragma unroll
    for (uint32_t i{ 0u }; i < 23u; ++i)
    {
        keccak_f1600_round(state, i);
    }

    uint64_t tmp[5];

    // theta
    tmp[0] = xor5(state, 0u);
    tmp[1] = xor5(state, 1u);
    tmp[2] = xor5(state, 2u);
    tmp[3] = xor5(state, 3u);
    tmp[4] = xor5(state, 4u);

    state[0]  = state[0]  ^ tmp[4] ^ rol_u64(tmp[1], 1u);
    state[6]  = state[6]  ^ tmp[0] ^ rol_u64(tmp[2], 1u);
    state[12] = state[12] ^ tmp[1] ^ rol_u64(tmp[3], 1u);

    // rho
    state[1] = rol_u64(state[6],  44u);
    state[2] = rol_u64(state[12], 43u);

    //chi
    state[0] = state[0] ^ ((~state[1]) & state[2]);

    // iota
    state[0] ^= KECCAK_F1600_ROUND[23u];
}


__device__ __forceinline__
uint32_t mix_reduce(
    uint4 const* __restrict__ const dag,
    uint4& matrix,
    uint32_t const word,
    uint32_t const thread_lane_id)
{
    #pragma unroll
    for (uint32_t i{ 0u }; i < 16u; ++i)
    {
        uint32_t start_index;
        uint32_t const index_gap{ i * 4u };
        uint32_t const index_mix{ i & 7u };

        // TODO: Should be possible to use texture memory for DAG ?
        // TODO: Remove modulo by "Integer Division by Invariants" method
        {
            start_index = fnv1(index_gap ^ word, reg_load(matrix.x, index_mix, THREADS_PER_HASH));
            start_index %= d_dag_number_item;
            start_index *= 8u;
            fnv1(matrix, dag[start_index + thread_lane_id]);
        }
        {
            start_index = fnv1((index_gap + 1u) ^ word, reg_load(matrix.y, index_mix, THREADS_PER_HASH));
            start_index %= d_dag_number_item;
            start_index *= 8u;
            fnv1(matrix, dag[start_index + thread_lane_id]);
        }
        {
            start_index = fnv1((index_gap + 2u) ^ word, reg_load(matrix.z, index_mix, THREADS_PER_HASH));
            start_index %= d_dag_number_item;
            start_index *= 8u;
            fnv1(matrix, dag[start_index + thread_lane_id]);
        }
        {
            start_index = fnv1((index_gap + 3u) ^ word, reg_load(matrix.w, index_mix, THREADS_PER_HASH));
            start_index %= d_dag_number_item;
            start_index *= 8u;
            fnv1(matrix, dag[start_index + thread_lane_id]);
        }
    }

    return fnv1_reduce(matrix);
}


__device__ __forceinline__
void ethash_create_mix_hash(
    uint4 const* __restrict__ const dag,
    uint64_t* const __restrict__ state,
    uint4 const* const __restrict__ seed,
    uint32_t const thread_id)
{
    uint32_t const thread_lane_id{ thread_id & 7u };
    uint32_t const index_seed{ thread_lane_id & 3u };

    uint32_t word0;

    #pragma unroll
    for (uint32_t lane_id{ 0u }; lane_id < THREADS_PER_HASH; ++lane_id)
    {
        uint4 matrix;
        uint4 copy_matrix;

        #pragma unroll
        for (uint32_t i{ 0u }; i < 4u; ++i)
        {
            copy_matrix.x = reg_load(seed[i].x, lane_id, THREADS_PER_HASH);
            copy_matrix.y = reg_load(seed[i].y, lane_id, THREADS_PER_HASH);
            copy_matrix.z = reg_load(seed[i].z, lane_id, THREADS_PER_HASH);
            copy_matrix.w = reg_load(seed[i].w, lane_id, THREADS_PER_HASH);
            if (i == index_seed)
            {
                matrix = copy_matrix;
            }
            // TODO: Delete this if, we can load data before
            if (i == 0)
            {
                word0 = copy_matrix.x;
            }
        }

        uint32_t const matrix_reduce{ mix_reduce(dag, matrix, word0, thread_lane_id) };

        uint4 const shuffle_1
        {
            reg_load(matrix_reduce, 0, THREADS_PER_HASH),
            reg_load(matrix_reduce, 1, THREADS_PER_HASH),
            reg_load(matrix_reduce, 2, THREADS_PER_HASH),
            reg_load(matrix_reduce, 3, THREADS_PER_HASH)
        };

        uint4 const shuffle_2
        {
            reg_load(matrix_reduce, 4, THREADS_PER_HASH),
            reg_load(matrix_reduce, 5, THREADS_PER_HASH),
            reg_load(matrix_reduce, 6, THREADS_PER_HASH),
            reg_load(matrix_reduce, 7, THREADS_PER_HASH)
        };

        // TODO: Find way or loading can be do after
        if (lane_id == thread_lane_id)
        {
            toU64(state, 8u, shuffle_1);
            toU64(state, 10u, shuffle_2);
        }
    }
}


__device__ __forceinline__
void check_nonce(
    t_result* const result,
    uint64_t const state0,
    uint64_t const nonce)
{
    uint64_t const bytes = be_u64(state0);
    if (bytes <= 1ull)
    {
        result->found = true;
        uint32_t const index{ atomicAdd((uint32_t*)&result->count, 1) };
        if (index < 1)
        {
            result->nonce = nonce;
        }
    }
}


__global__
void kernel_ethash_lm1(
    t_result* __restrict__ const result,
    uint4 const* __restrict__ const dag,
    uint64_t const start_nonce)
{
    uint64_t state[25];
    uint4 seed[4];
    uint32_t const thread_id{ (blockIdx.x * blockDim.x) + threadIdx.x };
    uint64_t const nonce{ thread_id + start_nonce };

    keccak_f1600_first(state, seed, nonce);
    ethash_create_mix_hash(dag, state, seed, thread_id);
    keccak_f1600_final(state);
    check_nonce(result, state[0], nonce);
}


__host__
bool init_ethash_lm1(
    algo::hash256 const* header_hash,
    uint64_t const dag_number_item)
{
    uint4 const* header{ (uint4*)&header_hash };

    CUDA_ER(cudaMemcpyToSymbol(d_header, header, sizeof(uint4) * 2));
    CUDA_ER(cudaMemcpyToSymbol(d_dag_number_item, (void*)&dag_number_item, sizeof(uint32_t)));

    return true;
}


__host__
bool ethash_lm1(
    cudaStream_t stream,
    t_result* const result,
    algo::hash1024* const dag,
    uint32_t const blocks,
    uint32_t const threads)
{
    uint64_t const nonce{ 0ull };

    kernel_ethash_lm1<<<blocks, threads, 0, stream>>>
    (
        result,
        (uint4*)dag,
        nonce
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
