///////////////////////////////////////////////////////////////////////////////
#include <common/cast.hpp>
#include <benchmark/cuda/kernels.hpp>

///////////////////////////////////////////////////////////////////////////////
#include <benchmark/cuda/common/common.cuh>

///////////////////////////////////////////////////////////////////////////////
__constant__ uint4* d_dag;
__constant__ uint4 d_header[2];
__constant__ uint64_t d_boundary;
__constant__ uint32_t d_dag_number_item;

///////////////////////////////////////////////////////////////////////////////
#include <benchmark/cuda/ethash/ethash_keccak_f1600.cuh>

//////////////////////////////////////////////////j/////////////////////////////
constexpr uint32_t _PARALLEL_HASH{ 4u };
constexpr uint32_t ACCESSES{ 64u };
constexpr uint32_t THREADS_PER_HASH{ 8u }; // (128 / 16)


///////////////////////////////////////////////////////////////////////////////
union
{
    uint32_t words[128 / sizeof(uint32_t)];
    uint2    uint2s[128 / sizeof(uint2)];
    uint4    uint4s[128 / sizeof(uint4)];
} hash128_t;


__device__ __forceinline__
uint64_t cuda_swab64(uint64_t const x)
{
    uint64_t result;
    uint2 t;
    asm volatile(
        "mov.b64 {%0,%1},%2; \n\t"
        : "=r"(t.x), "=r"(t.y)
        : "l"(x));
    t.x = __byte_perm(t.x, 0, 0x0123);
    t.y = __byte_perm(t.y, 0, 0x0123);
    asm volatile(
        "mov.b64 %0,{%1,%2}; \n\t"
        : "=l"(result) : "r"(t.y), "r"(t.x));
    return result;
}


__global__
void kernel_ethash_ethminer(
    volatile t_result_64* result,
    uint64_t const startNonce)
{
    uint2 state[12];

    uint32_t const gid{ (blockIdx.x * blockDim.x) + threadIdx.x };
    uint32_t const thread_id = threadIdx.x & (THREADS_PER_HASH - 1u);
    uint32_t const mix_idx = thread_id & 3;
    uint64_t const nonce = startNonce + gid;

    state[4] = vectorize(nonce);
    ethash_keccak_f1600_init(state);

    for (int i = 0; i < THREADS_PER_HASH; i += _PARALLEL_HASH)
    {
        uint4 mix[_PARALLEL_HASH];
        uint32_t offset[_PARALLEL_HASH];
        uint32_t init0[_PARALLEL_HASH];

        for (int p = 0; p < _PARALLEL_HASH; p++)
        {
            uint2 shuffle[8];
            for (int j = 0; j < 8; j++)
            {
                shuffle[j].x = reg_load(state[j].x, i + p, THREADS_PER_HASH);
                shuffle[j].y = reg_load(state[j].y, i + p, THREADS_PER_HASH);
            }

            switch (mix_idx)
            {
                case 0u:
                    mix[p] = vectorize_u2(shuffle[0], shuffle[1]);
                    break;
                case 1u:
                    mix[p] = vectorize_u2(shuffle[2], shuffle[3]);
                    break;
                case 2u:
                    mix[p] = vectorize_u2(shuffle[4], shuffle[5]);
                    break;
                case 3u:
                    mix[p] = vectorize_u2(shuffle[6], shuffle[7]);
                    break;
            }

            init0[p] = reg_load(shuffle[0].x, 0, THREADS_PER_HASH);
        }

        for (uint32_t a = 0u; a < ACCESSES; a += 4u)
        {
            int t = bfe(a, 2u, 3u);

            for (uint32_t b = 0u; b < 4u; b++)
            {
                for (int p = 0u; p < _PARALLEL_HASH; p++)
                {
                    offset[p] = fnv1(
                        init0[p] ^ (a + b),
                        ((uint32_t*)&mix[p])[b]) % d_dag_number_item;
                    offset[p] = reg_load(offset[p], t, THREADS_PER_HASH);
                    uint32_t start_index = offset[p];
                    fnv1(&mix[p], &d_dag[start_index * 8  + thread_id]);
                }
            }
        }

        for (uint32_t p = 0u; p < _PARALLEL_HASH; p++)
        {
            uint2 shuffle[4];
            uint32_t thread_mix = fnv1_reduce(mix[p]);

            shuffle[0].x = reg_load(thread_mix, 0, THREADS_PER_HASH);
            shuffle[0].y = reg_load(thread_mix, 1, THREADS_PER_HASH);
            shuffle[1].x = reg_load(thread_mix, 2, THREADS_PER_HASH);
            shuffle[1].y = reg_load(thread_mix, 3, THREADS_PER_HASH);
            shuffle[2].x = reg_load(thread_mix, 4, THREADS_PER_HASH);
            shuffle[2].y = reg_load(thread_mix, 5, THREADS_PER_HASH);
            shuffle[3].x = reg_load(thread_mix, 6, THREADS_PER_HASH);
            shuffle[3].y = reg_load(thread_mix, 7, THREADS_PER_HASH);

            if ((i + p) == thread_id)
            {
                state[8] = shuffle[0];
                state[9] = shuffle[1];
                state[10] = shuffle[2];
                state[11] = shuffle[3];
            }
        }
    }

    uint64_t const final_state = ethash_keccak_f1600_final(state);

    if (cuda_swab64(final_state) > d_boundary)
    {
        uint32_t const index = atomicAdd((uint32_t*)&result->index, 1);
        if (index >= MAX_RESULT_INDEX)
        {
            return;
        }

        result->found = true;
        result->nonce[index] = gid;
        result->mix[index][0] = state[8].x;
        result->mix[index][1] = state[8].y;
        result->mix[index][2] = state[9].x;
        result->mix[index][3] = state[9].y;
        result->mix[index][4] = state[10].x;
        result->mix[index][5] = state[10].y;
        result->mix[index][6] = state[11].x;
        result->mix[index][7] = state[11].y;
    }
}


__host__
bool init_ethash_ethminer(
    algo::hash1024 const* dagHash,
    algo::hash256 const* headerHash,
    uint64_t const dagNumberItem,
    uint64_t const boundary)
{
    uint4 const* header{ (uint4*)&headerHash };
    uint4 const* dag{ (uint4*)dagHash };

    CUDA_ER(cudaMemcpyToSymbol(d_header, header, sizeof(uint4) * 2));
    CUDA_ER(cudaMemcpyToSymbol(d_boundary, (void*)&boundary, sizeof(uint64_t)));
    CUDA_ER(cudaMemcpyToSymbol(d_dag_number_item, (void*)&dagNumberItem, sizeof(uint32_t)));
    CUDA_ER(cudaMemcpyToSymbol(d_dag, &dag, sizeof(uint4*)));

    return true;
}

__host__
bool ethash_ethminer(
        cudaStream_t stream,
        t_result_64* result,
        uint32_t const blocks,
        uint32_t const threads)
{
    kernel_ethash_ethminer<<<blocks, threads, 0, stream>>>
    (
        result,
        0x3835000000000000ull
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
