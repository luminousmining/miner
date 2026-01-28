#pragma once

#include <common/cuda/rotate_byte.cuh>
#include <common/cuda/to_u4.cuh>
#include <common/cuda/to_u32.cuh>
#include <common/cuda/to_u64.cuh>
#include <common/cuda/xor.cuh>


__device__ __constant__
uint64_t KECCAK_F1600_ROUND[24] =
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
void thetha(
    uint64_t* __restrict__ const state,
    uint64_t* __restrict__ const C,
    uint64_t* __restrict__ const D,
    uint64_t* __restrict__ const tmp)
{
    uint64_t value;

    C[0] = xor5(state, 0u);
    C[1] = xor5(state, 1u);
    C[2] = xor5(state, 2u);
    C[3] = xor5(state, 3u);
    C[4] = xor5(state, 4u);

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

    tmp[0]  = rol_u64(state[1],  1u);
    tmp[1]  = rol_u64(state[2],  62u);
    tmp[2]  = rol_u64(state[3],  28u);
    tmp[3]  = rol_u64(state[4],  27u);
    tmp[4]  = rol_u64(state[5],  36u);
    tmp[5]  = rol_u64(state[6],  44u);
    tmp[6]  = rol_u64(state[7],  6u);
    tmp[7]  = rol_u64(state[8],  55u);
    tmp[8]  = rol_u64(state[9],  20u);
    tmp[9] = rol_u64(state[10], 3u);
    tmp[10] = rol_u64(state[11], 10u);
    tmp[11] = rol_u64(state[12], 43u);
    tmp[12] = rol_u64(state[13], 25u);
    tmp[13] = rol_u64(state[14], 39u);
    tmp[14] = rol_u64(state[15], 41u);
    tmp[15] = rol_u64(state[16], 45u);
    tmp[16] = rol_u64(state[17], 15u);
    tmp[17] = rol_u64(state[18], 21u);
    tmp[18] = rol_u64(state[19], 8u);
    tmp[19] = rol_u64(state[20], 18u);
    tmp[20] = rol_u64(state[21], 2u);
    tmp[21] = rol_u64(state[22], 61u);
    tmp[22] = rol_u64(state[23], 56u);
    tmp[23] = rol_u64(state[24], 14u);
}


__device__ __forceinline__
void pi(
    uint64_t* __restrict__ const state,
    uint64_t* __restrict__ const tmp)
{
    state[16] = tmp[4];
    state[7] = tmp[9];
    state[23] = tmp[14];
    state[14] = tmp[19];

    state[10] = tmp[0];
    state[1] = tmp[5];
    state[17] = tmp[10];
    state[8] = tmp[15];
    state[24] = tmp[20];

    state[20] = tmp[1];
    state[11] = tmp[6];
    state[2] = tmp[11];
    state[18] = tmp[16];
    state[9] = tmp[21];

    state[5] = tmp[2];
    state[21] = tmp[7];
    state[12] = tmp[12];
    state[3] = tmp[17];
    state[19] = tmp[22];

    state[15] = tmp[3];
    state[6] = tmp[8];
    state[22] = tmp[13];
    state[13] = tmp[18];
    state[4] = tmp[23];
}


__device__ __forceinline__
void chi(
    uint64_t* __restrict__ const state,
    uint64_t* __restrict__ const C)
{
    #pragma unroll
    for (uint32_t i{ 0u }; i < 5u; ++i)
    {
        uint32_t const j{ i * 5u };
        C[0] = state[j]      ^ ((~state[j + 1u]) & state[j + 2u]);
        C[1] = state[j + 1u] ^ ((~state[j + 2u]) & state[j + 3u]);
        C[2] = state[j + 2u] ^ ((~state[j + 3u]) & state[j + 4u]);
        C[3] = state[j + 3u] ^ ((~state[j + 4u]) & state[j]);
        C[4] = state[j + 4u] ^ ((~state[j])      & state[j + 1u]);

        state[j]      = C[0];
        state[j + 1u] = C[1];
        state[j + 2u] = C[2];
        state[j + 3u] = C[3];
        state[j + 4u] = C[4];
    }
}


__device__ __forceinline__
void iota(
    uint64_t* const state,
    uint32_t const round)
{
    state[0] ^= KECCAK_F1600_ROUND[round];
}


__device__ __forceinline__
void keccak_f1600_round(
    uint64_t* state,
    uint32_t round)
{
    uint64_t C[5];
    uint64_t D[5];
    uint64_t tmp[24];

    thetha(state, C, D, tmp);
    pi(state, tmp);
    chi(state, C);
    iota(state, round);
}


__device__ __forceinline__
void keccak_process(
    uint64_t* const state)
{
    state[8] = 0x8000000000000001ull;

    #pragma unroll
    for (uint32_t i{ 9u }; i < 25u; ++i)
    {
        state[i] = 0ull;
    }

    #pragma unroll
    for (uint32_t i{ 0u }; i < 24u; ++i)
    {
        keccak_f1600_round(state, i);
    }
}


__device__ __forceinline__
void keccak_f1600(
    uint4* const hash)
{
    uint64_t state[25];

    toU64(state, 0, hash[0]);
    toU64(state, 2, hash[1]);
    toU64(state, 4, hash[2]);
    toU64(state, 6, hash[3]);

    keccak_process(state);

    hash[0] = toU4(state[0], state[1]);
    hash[1] = toU4(state[2], state[3]);
    hash[2] = toU4(state[4], state[5]);
    hash[3] = toU4(state[6], state[7]);
}


__device__ __forceinline__
void keccak_f1600_u32(
    uint32_t* const hash)
{
    uint64_t state[25];

    toU64FromU32(state, 0, hash);
    toU64FromU32(state, 2, hash + 4);
    toU64FromU32(state, 4, hash + 8);
    toU64FromU32(state, 6, hash + 12);

    keccak_process(state);

    toU32FromU64(hash, state[0], state[1]);
    toU32FromU64(hash + 4, state[2], state[3]);
    toU32FromU64(hash + 8, state[4], state[5]);
    toU32FromU64(hash + 12, state[6], state[7]);
}


__device__ __forceinline__
void keccak_f1600_u64(
    uint64_t* const hash)
{
    uint64_t state[25];

    state[0] = hash[0];
    state[1] = hash[1];
    state[2] = hash[2];
    state[3] = hash[3];
    state[4] = hash[4];
    state[5] = hash[5];
    state[6] = hash[6];
    state[7] = hash[7];

    keccak_process(state);

    hash[0] = state[0];
    hash[1] = state[1];
    hash[2] = state[2];
    hash[3] = state[3];
    hash[4] = state[4];
    hash[5] = state[5];
    hash[6] = state[6];
    hash[7] = state[7];
}
