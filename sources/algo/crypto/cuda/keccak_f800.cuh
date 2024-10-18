#pragma once

#if !defined(__KERNEL_COMPILED)
#include <common/cuda/rotate_byte.cuh>
#include <common/cuda/to_u4.cuh>
#include <common/cuda/to_u64.cuh>
#include <common/cuda/xor.cuh>
#endif


__device__ __forceinline__
void keccak_f800_round(
    uint32_t* const __restrict__ out)
{
    uint32_t tmp;
    uint32_t bc[5];
    uint32_t out_base[25]; //OUT_INDEX_MAX

    // Theta
    bc[0] = xor5(out, 0u);
    bc[1] = xor5(out, 1u);
    bc[2] = xor5(out, 2u);
    bc[3] = xor5(out, 3u);
    bc[4] = xor5(out, 4u);

    tmp = bc[4] ^ rol_u32(bc[1], 1u);
    out[0]  ^= tmp;
    out[5]  ^= tmp;
    out[10] ^= tmp;
    out[15] ^= tmp;
    out[20] ^= tmp;

    tmp = bc[0] ^ rol_u32(bc[2], 1u);
    out[1]  ^= tmp;
    out[6]  ^= tmp;
    out[11] ^= tmp;
    out[16] ^= tmp;
    out[21] ^= tmp;

    tmp = bc[1] ^ rol_u32(bc[3], 1u);
    out[2]  ^= tmp;
    out[7]  ^= tmp;
    out[12] ^= tmp;
    out[17] ^= tmp;
    out[22] ^= tmp;

    tmp = bc[2] ^ rol_u32(bc[4], 1u);
    out[3]  ^= tmp;
    out[8]  ^= tmp;
    out[13] ^= tmp;
    out[18] ^= tmp;
    out[23] ^= tmp;

    tmp = bc[3] ^ rol_u32(bc[0], 1u);
    out[4]  ^= tmp;
    out[9]  ^= tmp;
    out[14] ^= tmp;
    out[19] ^= tmp;
    out[24] ^= tmp;

    // Rho PI
    #pragma unroll
    for (uint32_t i = 0u; i < 25u; ++i)
    {
        out_base[i] = out[i];
    }
    out[10] = rol_u32(out_base[1], 1u);
    out[7]  = rol_u32(out_base[10], 3u);
    out[11] = rol_u32(out_base[7], 6u);
    out[17] = rol_u32(out_base[11], 10u);
    out[18] = rol_u32(out_base[17], 15u);
    out[3]  = rol_u32(out_base[18], 21u);
    out[5]  = rol_u32(out_base[3], 28u);
    out[16] = rol_u32(out_base[5], 36u);
    out[8]  = rol_u32(out_base[16], 45u);
    out[21] = rol_u32(out_base[8], 55u);
    out[24] = rol_u32(out_base[21], 2u);
    out[4]  = rol_u32(out_base[24], 14u);
    out[15] = rol_u32(out_base[4], 27u);
    out[23] = rol_u32(out_base[15], 41u);
    out[19] = rol_u32(out_base[23], 56u);
    out[13] = rol_u32(out_base[19], 8u);
    out[12] = rol_u32(out_base[13], 25u);
    out[2]  = rol_u32(out_base[12], 43u);
    out[20] = rol_u32(out_base[2], 62);
    out[14] = rol_u32(out_base[20], 18u);
    out[22] = rol_u32(out_base[14], 39u);
    out[9]  = rol_u32(out_base[22], 61u);
    out[6]  = rol_u32(out_base[9], 20u);
    out[1]  = rol_u32(out_base[6], 44u);

    // Chi
    #pragma unroll
    for (uint32_t i{ 0u }; i < 5u; ++i)
    {
        tmp = i * 5u ;
        bc[0] = out[tmp]     ^ ((~out[tmp + 1]) & out[tmp + 2]);
        bc[1] = out[tmp + 1] ^ ((~out[tmp + 2]) & out[tmp + 3]);
        bc[2] = out[tmp + 2] ^ ((~out[tmp + 3]) & out[tmp + 4]);
        bc[3] = out[tmp + 3] ^ ((~out[tmp + 4]) & out[tmp]);
        bc[4] = out[tmp + 4] ^ ((~out[tmp])     & out[tmp + 1]);

        out[tmp]     = bc[0];
        out[tmp + 1] = bc[1];
        out[tmp + 2] = bc[2];
        out[tmp + 3] = bc[3];
        out[tmp + 4] = bc[4];
    }
}


__device__ __forceinline__
void keccak_f800(
    uint32_t* const __restrict__ flat_matrix)
{
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x00000001u;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x00008082u;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x0000808au;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x80008000u;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x0000808bu;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x80000001u;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x80008081u;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x00008009u;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x0000008au;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x00000088u;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x80008009u;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x8000000au;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x8000808bu;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x0000008bu;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x00008089u;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x00008003u;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x00008002u;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x00000080u;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x0000800au;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x8000000au;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x80008081u;
    keccak_f800_round(flat_matrix); flat_matrix[0] ^= 0x00008080u;
}
