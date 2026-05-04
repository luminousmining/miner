///////////////////////////////////////////////////////////////////////////////
#include <common/cast.hpp>
#include <benchmark/result.hpp>

///////////////////////////////////////////////////////////////////////////////
#include <benchmark/cuda/common/common.cuh>

///////////////////////////////////////////////////////////////////////////////
#include <common/cuda/be_u64.cuh>
#include <common/cuda/fast_mod.cuh>
#include <common/cuda/register.cuh>

///////////////////////////////////////////////////////////////////////////////
#include <algo/crypto/cuda/fnv1.cuh>

///////////////////////////////////////////////////////////////////////////////
__constant__ uint4 d_header[2];
__constant__ FastDivisor d_dag_divisor;

///////////////////////////////////////////////////////////////////////////////
// Ethminer Keccak: uint2 state + lop3 PTX
#include <benchmark/cuda/ethash/ethash_keccak_f1600.cuh>

///////////////////////////////////////////////////////////////////////////////
constexpr uint32_t THREADS_PER_HASH{ 8u };
constexpr uint32_t PARALLEL_HASH{ 4u };
constexpr uint32_t ACCESSES{ 64u };


// Combined mix hash: PARALLEL_HASH=4 + fast_mod + uint2 keccak state output
__device__ __forceinline__
void ethash_create_mix_hash_lm6(
    uint4 const* __restrict__ const dag,
    uint2* const __restrict__ keccak_state,
    uint4 const* const __restrict__ seed,
    uint32_t const thread_id)
{
    uint32_t const thread_lane_id{ thread_id & 7u };
    uint32_t const mix_idx{ thread_lane_id & 3u };

    #pragma unroll 1
    for (uint32_t lane_base{ 0u }; lane_base < THREADS_PER_HASH; lane_base += PARALLEL_HASH)
    {
        uint4    mix[PARALLEL_HASH];
        uint32_t init0[PARALLEL_HASH];

        // Load PARALLEL_HASH initial mix states from PARALLEL_HASH different nonces
        #pragma unroll
        for (uint32_t p{ 0u }; p < PARALLEL_HASH; ++p)
        {
            uint32_t const lane_id{ lane_base + p };

            uint4 s0, s1, s2, s3;
            s0.x = reg_load(seed[0].x, lane_id, THREADS_PER_HASH);
            s0.y = reg_load(seed[0].y, lane_id, THREADS_PER_HASH);
            s0.z = reg_load(seed[0].z, lane_id, THREADS_PER_HASH);
            s0.w = reg_load(seed[0].w, lane_id, THREADS_PER_HASH);
            s1.x = reg_load(seed[1].x, lane_id, THREADS_PER_HASH);
            s1.y = reg_load(seed[1].y, lane_id, THREADS_PER_HASH);
            s1.z = reg_load(seed[1].z, lane_id, THREADS_PER_HASH);
            s1.w = reg_load(seed[1].w, lane_id, THREADS_PER_HASH);
            s2.x = reg_load(seed[2].x, lane_id, THREADS_PER_HASH);
            s2.y = reg_load(seed[2].y, lane_id, THREADS_PER_HASH);
            s2.z = reg_load(seed[2].z, lane_id, THREADS_PER_HASH);
            s2.w = reg_load(seed[2].w, lane_id, THREADS_PER_HASH);
            s3.x = reg_load(seed[3].x, lane_id, THREADS_PER_HASH);
            s3.y = reg_load(seed[3].y, lane_id, THREADS_PER_HASH);
            s3.z = reg_load(seed[3].z, lane_id, THREADS_PER_HASH);
            s3.w = reg_load(seed[3].w, lane_id, THREADS_PER_HASH);

            switch (mix_idx)
            {
                case 0u: mix[p] = s0; break;
                case 1u: mix[p] = s1; break;
                case 2u: mix[p] = s2; break;
                case 3u: mix[p] = s3; break;
            }

            init0[p] = s0.x;
        }

        // 64 DAG accesses per nonce — PARALLEL_HASH nonces simultaneously — fast_mod
        #pragma unroll 1
        for (uint32_t a{ 0u }; a < ACCESSES; a += 4u)
        {
            uint32_t const t{ (a >> 2u) & 7u };

            for (uint32_t b{ 0u }; b < 4u; ++b)
            {
                #pragma unroll
                for (uint32_t p{ 0u }; p < PARALLEL_HASH; ++p)
                {
                    uint32_t start_index{ fnv1((a + b) ^ init0[p], ((uint32_t*)&mix[p])[b]) };
                    start_index = fast_mod(d_dag_divisor, start_index);
                    start_index = reg_load(start_index, t, THREADS_PER_HASH);
                    start_index *= 8u;
                    fnv1(mix[p], dag[start_index + thread_lane_id]);
                }
            }
        }

        // Reduce and store results into uint2 keccak state
        #pragma unroll
        for (uint32_t p{ 0u }; p < PARALLEL_HASH; ++p)
        {
            uint32_t const matrix_reduce{ fnv1_reduce(mix[p]) };

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

            if ((lane_base + p) == thread_lane_id)
            {
                keccak_state[8]  = make_uint2(shuffle_1.x, shuffle_1.y);
                keccak_state[9]  = make_uint2(shuffle_1.z, shuffle_1.w);
                keccak_state[10] = make_uint2(shuffle_2.x, shuffle_2.y);
                keccak_state[11] = make_uint2(shuffle_2.z, shuffle_2.w);
            }
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
void kernel_ethash_lm6(
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

    // Mix hash: PARALLEL_HASH=4 + fast_mod, writes result into keccak_state[8..11]
    ethash_create_mix_hash_lm6(dag, keccak_state, seed, thread_id);

    // Final Keccak (ethminer style, returns uint64_t hash directly)
    uint64_t const final_hash{ ethash_keccak_f1600_final(keccak_state) };
    check_nonce(result, final_hash, nonce);
}


__host__
bool init_ethash_lm6(
    algo::hash256 const* header_hash,
    uint64_t const dag_number_item)
{
    uint4 const* header{ (uint4*)&header_hash };
    FastDivisor const divisor{ initFastMod(static_cast<uint32_t>(dag_number_item)) };

    CUDA_ER(cudaMemcpyToSymbol(d_header, header, sizeof(uint4) * 2));
    CUDA_ER(cudaMemcpyToSymbol(d_dag_divisor, (void*)&divisor, sizeof(FastDivisor)));

    return true;
}


__host__
bool ethash_lm6(
    cudaStream_t stream,
    t_result* const result,
    algo::hash1024* const dag,
    uint32_t const blocks,
    uint32_t const threads)
{
    uint64_t const nonce{ 0ull };

    kernel_ethash_lm6<<<blocks, threads, 0, stream>>>
    (
        result,
        (uint4*)dag,
        nonce
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
