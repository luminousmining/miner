#pragma once

#define SEARCH_PARRALLEL_LANE 8u


__device__ __forceinline__
void keccak_f1600_first(
    uint64_t* __restrict__ const state,
    uint4* __restrict__ const seed,
    uint4 const* __restrict__ const header,
    uint64_t const nonce)
{
    toU64(state, 0u, header[0]);
    toU64(state, 2u, header[1]);

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

    // On reduit le tableau en uint4
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
            start_index = fnv1(index_gap ^ word, __shfl_sync(0xffffffff, matrix.x, index_mix, SEARCH_PARRALLEL_LANE));
            start_index %= d_dag_number_item;
            start_index *= 8u;
            fnv1(matrix, d_dag[start_index + thread_lane_id]);
        }
        {
            start_index = fnv1((index_gap + 1u) ^ word, __shfl_sync(0xffffffff, matrix.y, index_mix, SEARCH_PARRALLEL_LANE));
            start_index %= d_dag_number_item;
            start_index *= 8u;
            fnv1(matrix, d_dag[start_index + thread_lane_id]);
        }
        {
            start_index = fnv1((index_gap + 2u) ^ word, __shfl_sync(0xffffffff, matrix.z, index_mix, SEARCH_PARRALLEL_LANE));
            start_index %= d_dag_number_item;
            start_index *= 8u;
            fnv1(matrix, d_dag[start_index + thread_lane_id]);
        }
        {
            start_index = fnv1((index_gap + 3u) ^ word, __shfl_sync(0xffffffff, matrix.w, index_mix, SEARCH_PARRALLEL_LANE));
            start_index %= d_dag_number_item;
            start_index *= 8u;
            fnv1(matrix, d_dag[start_index + thread_lane_id]);
        }
    }

    return fnv1_reduce(matrix);
}


__device__ __forceinline__
void ethash_create_mix_hash(
    uint64_t* const __restrict__ state,
    uint4 const* const __restrict__ seed,
    uint32_t const thread_id)
{
    uint32_t const thread_lane_id{ thread_id & 7u };
    uint32_t const index_seed{ thread_lane_id & 3u };

    uint32_t word0;

    #pragma unroll
    for (uint32_t lane_id{ 0u }; lane_id < SEARCH_PARRALLEL_LANE; ++lane_id)
    {
        uint4 matrix;
        uint4 copy_matrix;

        #pragma unroll
        for (uint32_t i{ 0u }; i < 4u; ++i)
        {
            copy_matrix.x = __shfl_sync(0xffffffff, seed[i].x, lane_id, SEARCH_PARRALLEL_LANE);
            copy_matrix.y = __shfl_sync(0xffffffff, seed[i].y, lane_id, SEARCH_PARRALLEL_LANE);
            copy_matrix.z = __shfl_sync(0xffffffff, seed[i].z, lane_id, SEARCH_PARRALLEL_LANE);
            copy_matrix.w = __shfl_sync(0xffffffff, seed[i].w, lane_id, SEARCH_PARRALLEL_LANE);
            if (i == index_seed)
            {
                matrix = copy_matrix;
            }
            if (i == 0)
            {
                word0 = copy_matrix.x;
            }
        }

        uint32_t const matrix_reduce{ mix_reduce(matrix, word0, thread_lane_id) };

        uint4 const shuffle_1
        {
            __shfl_sync(0xffffffff, matrix_reduce, 0, SEARCH_PARRALLEL_LANE),
            __shfl_sync(0xffffffff, matrix_reduce, 1, SEARCH_PARRALLEL_LANE),
            __shfl_sync(0xffffffff, matrix_reduce, 2, SEARCH_PARRALLEL_LANE),
            __shfl_sync(0xffffffff, matrix_reduce, 3, SEARCH_PARRALLEL_LANE)
        };

        uint4 const shuffle_2
        {
            __shfl_sync(0xffffffff, matrix_reduce, 4, SEARCH_PARRALLEL_LANE),
            __shfl_sync(0xffffffff, matrix_reduce, 5, SEARCH_PARRALLEL_LANE),
            __shfl_sync(0xffffffff, matrix_reduce, 6, SEARCH_PARRALLEL_LANE),
            __shfl_sync(0xffffffff, matrix_reduce, 7, SEARCH_PARRALLEL_LANE)
        };

        if (lane_id == thread_lane_id)
        {
            toU64(state, 8u, shuffle_1);
            toU64(state, 10u, shuffle_2);
        }
    }
}


__device__ __forceinline__
void check_nonce(
    algo::ethash::Result* const result,
    uint64_t const state0,
    uint64_t const nonce)
{
    if (be_u64(state0) <= d_boundary)
    {
        result->found = true;
        uint32_t const index{ atomicAdd((uint32_t* const)&result->count, 1) };
        if (4u > index)
        {
            result->nonces[index] = nonce;
        }
    }
}


__global__
void search(
    algo::ethash::Result* const result,
    uint64_t const startNonce)
{
    uint64_t state[25];
    uint4 seed[4];
    uint32_t const thread_id{ (blockIdx.x * blockDim.x) + threadIdx.x };
    uint64_t const nonce{ thread_id + startNonce };

    keccak_f1600_first(state, seed, d_header, nonce);
    ethash_create_mix_hash(state, seed, thread_id);
    keccak_f1600_final(state);
    check_nonce(result, state[0], nonce);
}


__host__
bool ethashSearch(
    cudaStream_t stream,
    algo::ethash::Result* const result,
    uint32_t const blocks,
    uint32_t const threads,
    uint64_t const startNonce)
{
    search<<<blocks, threads, 0, stream>>>(result, startNonce);
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
