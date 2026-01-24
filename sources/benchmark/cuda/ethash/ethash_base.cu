///////////////////////////////////////////////////////////////////////////////
#include <common/cast.hpp>
#include <algo/hash.hpp>
#include <benchmark/result.hpp>

///////////////////////////////////////////////////////////////////////////////
#include <benchmark/cuda/common/common.cuh>

///////////////////////////////////////////////////////////////////////////////
#include <common/cuda/be_u64.cuh>
#include <common/cuda/register.cuh>

///////////////////////////////////////////////////////////////////////////////
#include <algo/crypto/cuda/fnv1.cuh>

///////////////////////////////////////////////////////////////////////////////
__constant__ uint64_t d_header[8];
__constant__ uint32_t d_dag_number_item;


__device__ __constant__
uint64_t KECCAK_ROUND_ETHASH_BASE[24] =
{
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};


__device__ __forceinline__
void keccak_f1600_ethash_last_round(
    uint64_t* __restrict__ const state)
{
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

    state[0] ^= D[1] ^ C[4];
    state[10] ^= D[1] ^ C[4];

    state[6] ^= D[2] ^ C[0];
    state[16] ^= D[2] ^ C[0];

    state[12] ^= D[3] ^ C[1];
    state[22] ^= D[3] ^ C[1];

    state[3] ^= D[4] ^ C[2];
    state[18] ^= D[4] ^ C[2];

    state[9] ^= D[0] ^ C[3];
    state[24] ^= D[0] ^ C[3];

    // rho pi
    state[1] = rol_u64(state[6], 44u);
    state[6] = rol_u64(state[9], 20u);
    state[9] = rol_u64(state[22], 61u);
    state[2] = rol_u64(state[12], 43u);
    state[4] = rol_u64(state[24], 14u);
    state[8] = rol_u64(state[16], 45u);
    state[5] = rol_u64(state[3], 28u);
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
    state[0] ^= KECCAK_ROUND_ETHASH_BASE[23u];
}


__device__ __forceinline__
void keccak_f1600_ethash_final_hash(
    uint64_t* __restrict__ const state)
{
    uint64_t t[5];

    // theta
    t[0] = xor5(state, 0);
    t[1] = xor5(state, 1);
    t[2] = xor5(state, 2);
    t[3] = xor5(state, 3);
    t[4] = xor5(state, 4);

    state[0] = state[0] ^ t[4] ^ rol_u64(t[1], 1u);
    state[6] = state[6] ^ t[0] ^ rol_u64(t[2], 1u);
    state[12] = state[12] ^ t[1] ^ rol_u64(t[3], 1u);

    // rho
    state[1] = rol_u64(state[6], 44u);
    state[2] = rol_u64(state[12], 43u);

    //chi
    state[0] = state[0] ^ ((~state[1]) & state[2]);

    // iota
    state[0] ^= KECCAK_ROUND_ETHASH_BASE[23u];
}


__device__ __forceinline__
void keccak_f1600_round(
    uint64_t* const __restrict__ state,
    uint32_t const round)
{
    uint64_t value;
    uint64_t C[5];
    uint64_t D[5];
    uint64_t tmp[25];

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

    value = D[1] ^ C[4];
    state[0] ^= value;
    state[5] ^= value;
    state[10] ^= value;
    state[15] ^= value;
    state[20] ^= value;

    value = D[2] ^ C[0];
    state[1] ^= value;
    state[6] ^= value;
    state[11] ^= value;
    state[16] ^= value;
    state[21] ^= value;

    value = D[3] ^ C[1];
    state[2] ^= value;
    state[7] ^= value;
    state[12] ^= value;
    state[17] ^= value;
    state[22] ^= value;

    value = D[4] ^ C[2];
    state[3] ^= value;
    state[8] ^= value;
    state[13] ^= value;
    state[18] ^= value;
    state[23] ^= value;

    value = D[0] ^ C[3];
    state[4] ^= value;
    state[9] ^= value;
    state[14] ^= value;
    state[19] ^= value;
    state[24] ^= value;

    tmp[1] = rol_u64(state[1], 1u);
    tmp[2] = rol_u64(state[2], 62u);
    tmp[3] = rol_u64(state[3], 28u);
    tmp[4] = rol_u64(state[4], 27u);
    tmp[5] = rol_u64(state[5], 36u);
    tmp[6] = rol_u64(state[6], 44u);
    tmp[7] = rol_u64(state[7], 6u);
    tmp[8] = rol_u64(state[8], 55u);
    tmp[9] = rol_u64(state[9], 20u);
    tmp[10] = rol_u64(state[10], 3u);
    tmp[11] = rol_u64(state[11], 10u);
    tmp[12] = rol_u64(state[12], 43u);
    tmp[13] = rol_u64(state[13], 25u);
    tmp[14] = rol_u64(state[14], 39u);
    tmp[15] = rol_u64(state[15], 41u);
    tmp[16] = rol_u64(state[16], 45u);
    tmp[17] = rol_u64(state[17], 15u);
    tmp[18] = rol_u64(state[18], 21u);
    tmp[19] = rol_u64(state[19], 8u);
    tmp[20] = rol_u64(state[20], 18u);
    tmp[21] = rol_u64(state[21], 2u);
    tmp[22] = rol_u64(state[22], 61u);
    tmp[23] = rol_u64(state[23], 56u);
    tmp[24] = rol_u64(state[24], 14u);

    // PI
    state[0] = tmp[0];
    state[16] = tmp[5];
    state[7] = tmp[10];
    state[23] = tmp[15];
    state[14] = tmp[20];

    state[10] = tmp[1];
    state[1] = tmp[6];
    state[17] = tmp[11];
    state[8] = tmp[16];
    state[24] = tmp[21];

    state[20] = tmp[2];
    state[11] = tmp[7];
    state[2] = tmp[12];
    state[18] = tmp[17];
    state[9] = tmp[22];

    state[5] = tmp[3];
    state[21] = tmp[8];
    state[12] = tmp[13];
    state[3] = tmp[18];
    state[19] = tmp[23];

    state[15] = tmp[4];
    state[6] = tmp[9];
    state[22] = tmp[14];
    state[13] = tmp[19];
    state[4] = tmp[24];

    // CHI
    #pragma unroll
    for (uint32_t i{ 0u }; i < 5u; ++i)
    {
        uint32_t const j{ i * 5u };
        C[0] = state[j] ^ ((~state[j + 1]) & state[j + 2]);
        C[1] = state[j + 1] ^ ((~state[j + 2]) & state[j + 3]);
        C[2] = state[j + 2] ^ ((~state[j + 3]) & state[j + 4]);
        C[3] = state[j + 3] ^ ((~state[j + 4]) & state[j]);
        C[4] = state[j + 4] ^ ((~state[j]) & state[j + 1]);

        state[j] = C[0];
        state[j + 1] = C[1];
        state[j + 2] = C[2];
        state[j + 3] = C[3];
        state[j + 4] = C[4];
    }

    // IOTA
    state[0] ^= KECCAK_ROUND_ETHASH_BASE[round];
}


__device__ __forceinline__
void keccak_f1600(
    uint64_t* __restrict__ const state,
    uint32_t const start_index,
    uint32_t const total_round)
{
    #pragma unroll
    for (uint32_t round{ start_index }; round < total_round; ++round)
    {
        keccak_f1600_round(state, round);
    }
}


__device__ __forceinline__
void ethash_build_seed(
    uint64_t* const __restrict__ state,
    algo::hash512& seed,
    uint64_t nonce)
{
    #pragma unroll
    for (uint32_t i{ 0u }; i < static_cast<uint32_t>(algo::LEN_HASH_512_WORD_64); ++i)
    {
        state[i] = d_header[i];
    }
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

    keccak_f1600(state, 0u, 23u);
    keccak_f1600_ethash_last_round(state);

    #pragma unroll
    for (uint32_t i{ 0u }; i < static_cast<uint32_t>(algo::LEN_HASH_512_WORD_64); ++i)
    {
        seed.word64[i] = state[i];
    }
}


__device__ __forceinline__
void ethash_create_mix_hash(
    algo::hash1024 const* const __restrict__ dag,
    uint64_t* const __restrict__ state,
    algo::hash512 const& seed)
{
    algo::hash1024 mix{};
    mix.h512[0] = seed;
    mix.h512[1] = seed;

    uint32_t const seedWord0{ seed.word32[0] };

    for (uint32_t i{ 0u }; i < 64u; ++i)
    {
        uint32_t const u{ i ^ seedWord0 };
        uint32_t const indexMix{ i % static_cast<uint32_t>(algo::LEN_HASH_1024_WORD_32) };
        uint32_t const v{ mix.word32[indexMix] };
        uint32_t const index{ (fnv1(u, v) % d_dag_number_item) };

        algo::hash1024 const& item{ dag[index] };
        for (uint32_t j{ 0u }; j < static_cast<uint32_t>(algo::LEN_HASH_1024_WORD_32); ++j)
        {
            mix.word32[j] = fnv1(mix.word32[j], item.word32[j]);
        }
    }

    algo::hash256 mix_hash;
    #pragma unroll
    for (uint32_t i{ 0u }; i < static_cast<uint32_t>(algo::LEN_HASH_1024_WORD_32); i += 4u)
    {
        uint32_t h1{ fnv1(mix.word32[i], mix.word32[i + 1u]) };
        uint32_t h2{ fnv1(h1,            mix.word32[i + 2u]) };
        uint32_t h3{ fnv1(h2,            mix.word32[i + 3u]) };
        mix_hash.word32[i / 4u] = h3;
    }

    state[8] = mix_hash.word64[0];
    state[9] = mix_hash.word64[1];
    state[10] = mix_hash.word64[2];
    state[11] = mix_hash.word64[3];
}


__device__ __forceinline__
void ethash_final_hash(
    uint64_t* const __restrict__ state)
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

    keccak_f1600(state, 0u, 23u);
    keccak_f1600_ethash_final_hash(state);
}


__device__ __forceinline__
void check_nonce(
    t_result* __restrict__ const result,
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
void kernel_ethash_base(
    t_result* __restrict__ const result,
    algo::hash1024 const* const __restrict__ dag,
    uint64_t const start_nonce)
{
    uint64_t state[25];
    algo::hash512 seed{};
    uint32_t const thread_id{ (blockIdx.x * blockDim.x) + threadIdx.x };
    uint64_t const nonce{ thread_id + start_nonce };

    ethash_build_seed(state, seed, nonce);
    ethash_create_mix_hash(dag, state, seed);
    ethash_final_hash(state);
    check_nonce(result, state[0], nonce);
}


__host__
bool init_ethash_base(
    algo::hash256 const* header_hash,
    uint64_t const dag_number_item)
{
    uint4 const* header{ (uint4*)&header_hash };

    CUDA_ER(cudaMemcpyToSymbol(d_header, header, sizeof(uint4) * 2));
    CUDA_ER(cudaMemcpyToSymbol(d_dag_number_item, (void*)&dag_number_item, sizeof(uint32_t)));

    return true;
}


__host__
bool ethash_base(
    cudaStream_t stream,
    t_result* const result,
    algo::hash1024* const dag,
    uint32_t const blocks,
    uint32_t const threads)
{
    uint64_t const nonce{ 0ull };

    kernel_ethash_base<<<blocks, threads, 0, stream>>>
    (
        result,
        dag,
        nonce
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
