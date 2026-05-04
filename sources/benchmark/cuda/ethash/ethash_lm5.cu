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
// Ethminer Keccak: uint2 state + lop3 PTX (replaces keccak_f1600.cuh)
#include <benchmark/cuda/ethash/ethash_keccak_f1600.cuh>

///////////////////////////////////////////////////////////////////////////////
constexpr uint32_t THREADS_PER_HASH{ 8u };


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


// Adapted mix hash: reads seed from uint2 keccak state, writes result back as uint2
__device__ __forceinline__
void ethash_create_mix_hash_u2(
    uint4 const* __restrict__ const dag,
    uint2* const __restrict__ keccak_state,
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
            if (i == 0)
            {
                word0 = copy_matrix.x;
            }
        }

        uint32_t const matrix_reduce{ mix_reduce(dag, matrix, word0, thread_lane_id) };

        uint4 const shuffle_1
        {
            reg_load(matrix_reduce, 0u, THREADS_PER_HASH),
            reg_load(matrix_reduce, 1u, THREADS_PER_HASH),
            reg_load(matrix_reduce, 2u, THREADS_PER_HASH),
            reg_load(matrix_reduce, 3u, THREADS_PER_HASH)
        };

        uint4 const shuffle_2
        {
            reg_load(matrix_reduce, 4u, THREADS_PER_HASH),
            reg_load(matrix_reduce, 5u, THREADS_PER_HASH),
            reg_load(matrix_reduce, 6u, THREADS_PER_HASH),
            reg_load(matrix_reduce, 7u, THREADS_PER_HASH)
        };

        // Store into uint2 keccak state instead of uint64_t state
        if (lane_id == thread_lane_id)
        {
            keccak_state[8]  = make_uint2(shuffle_1.x, shuffle_1.y);
            keccak_state[9]  = make_uint2(shuffle_1.z, shuffle_1.w);
            keccak_state[10] = make_uint2(shuffle_2.x, shuffle_2.y);
            keccak_state[11] = make_uint2(shuffle_2.z, shuffle_2.w);
        }
    }
}


__device__ __forceinline__
void check_nonce(
    t_result* const result,
    uint64_t const state0,
    uint64_t const nonce)
{
    uint64_t const bytes{ be_u64(state0) };
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
void kernel_ethash_lm5(
    t_result* __restrict__ const result,
    uint4 const* __restrict__ const dag,
    uint64_t const start_nonce)
{
    uint32_t const thread_id{ (blockIdx.x * blockDim.x) + threadIdx.x };
    uint64_t const nonce{ thread_id + start_nonce };

    // Keccak state as uint2 (ethminer style: lop3 PTX optimized)
    uint2 keccak_state[12];
    keccak_state[4] = make_uint2(static_cast<uint32_t>(nonce), static_cast<uint32_t>(nonce >> 32u));
    ethash_keccak_f1600_init(keccak_state);

    // Convert keccak_state[0..7] (uint2) to seed[0..3] (uint4) for mix hash
    uint4 seed[4];
    seed[0] = make_uint4(keccak_state[0].x, keccak_state[0].y, keccak_state[1].x, keccak_state[1].y);
    seed[1] = make_uint4(keccak_state[2].x, keccak_state[2].y, keccak_state[3].x, keccak_state[3].y);
    seed[2] = make_uint4(keccak_state[4].x, keccak_state[4].y, keccak_state[5].x, keccak_state[5].y);
    seed[3] = make_uint4(keccak_state[6].x, keccak_state[6].y, keccak_state[7].x, keccak_state[7].y);

    // Mix hash (writes result into keccak_state[8..11])
    ethash_create_mix_hash_u2(dag, keccak_state, seed, thread_id);

    // Final Keccak (ethminer style, returns uint64_t hash directly)
    uint64_t const final_hash{ ethash_keccak_f1600_final(keccak_state) };
    check_nonce(result, final_hash, nonce);
}


__host__
bool init_ethash_lm5(
    algo::hash256 const* header_hash,
    uint64_t const dag_number_item)
{
    uint4 const* header{ (uint4*)&header_hash };

    CUDA_ER(cudaMemcpyToSymbol(d_header, header, sizeof(uint4) * 2));
    CUDA_ER(cudaMemcpyToSymbol(d_dag_number_item, (void*)&dag_number_item, sizeof(uint32_t)));

    return true;
}


__host__
bool ethash_lm5(
    cudaStream_t stream,
    t_result* const result,
    algo::hash1024* const dag,
    uint32_t const blocks,
    uint32_t const threads)
{
    uint64_t const nonce{ 0ull };

    kernel_ethash_lm5<<<blocks, threads, 0, stream>>>
    (
        result,
        (uint4*)dag,
        nonce
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
