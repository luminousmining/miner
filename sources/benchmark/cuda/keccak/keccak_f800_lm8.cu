///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
#include <common/cuda/rotate_byte.cuh>
#include <common/cuda/xor.cuh>

///////////////////////////////////////////////////////////////////////////////
#include <common/error/cuda_error.hpp>


__device__ __forceinline__
void keccak_f800_round_lm8(
    uint32_t* const __restrict__ out)
{
    uint32_t d0, d1, d2, d3, d4;
    uint32_t bc[5];
    uint32_t out_base[25];

    // Theta
    bc[0] = xor5(out, 0u);
    bc[1] = xor5(out, 1u);
    bc[2] = xor5(out, 2u);
    bc[3] = xor5(out, 3u);
    bc[4] = xor5(out, 4u);

    d0 = bc[4] ^ rol_u32(bc[1], 1u);
    d1 = bc[0] ^ rol_u32(bc[2], 1u);
    d2 = bc[1] ^ rol_u32(bc[3], 1u);
    d3 = bc[2] ^ rol_u32(bc[4], 1u);
    d4 = bc[3] ^ rol_u32(bc[0], 1u);

    out[0]  ^= d0;
    out[5]  ^= d0;
    out[10] ^= d0;
    out[15] ^= d0;
    out[20] ^= d0;

    out[1]  ^= d1;
    out[6]  ^= d1;
    out[11] ^= d1;
    out[16] ^= d1;
    out[21] ^= d1;

    out[2]  ^= d2;
    out[7]  ^= d2;
    out[12] ^= d2;
    out[17] ^= d2;
    out[22] ^= d2;

    out[3]  ^= d3;
    out[8]  ^= d3;
    out[13] ^= d3;
    out[18] ^= d3;
    out[23] ^= d3;

    out[4]  ^= d4;
    out[9]  ^= d4;
    out[14] ^= d4;
    out[19] ^= d4;
    out[24] ^= d4;

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

    // Chi - Manually unrolled
    bc[0] = out[0]  ^ ((~out[1])  & out[2]);
    bc[1] = out[1]  ^ ((~out[2])  & out[3]);
    bc[2] = out[2]  ^ ((~out[3])  & out[4]);
    bc[3] = out[3]  ^ ((~out[4])  & out[0]);
    bc[4] = out[4]  ^ ((~out[0])  & out[1]);
    out[0] = bc[0];
    out[1] = bc[1];
    out[2] = bc[2];
    out[3] = bc[3];
    out[4] = bc[4];

    bc[0] = out[5]  ^ ((~out[6])  & out[7]);
    bc[1] = out[6]  ^ ((~out[7])  & out[8]);
    bc[2] = out[7]  ^ ((~out[8])  & out[9]);
    bc[3] = out[8]  ^ ((~out[9])  & out[5]);
    bc[4] = out[9]  ^ ((~out[5])  & out[6]);
    out[5] = bc[0];
    out[6] = bc[1];
    out[7] = bc[2];
    out[8] = bc[3];
    out[9] = bc[4];

    bc[0] = out[10] ^ ((~out[11]) & out[12]);
    bc[1] = out[11] ^ ((~out[12]) & out[13]);
    bc[2] = out[12] ^ ((~out[13]) & out[14]);
    bc[3] = out[13] ^ ((~out[14]) & out[10]);
    bc[4] = out[14] ^ ((~out[10]) & out[11]);
    out[10] = bc[0];
    out[11] = bc[1];
    out[12] = bc[2];
    out[13] = bc[3];
    out[14] = bc[4];

    bc[0] = out[15] ^ ((~out[16]) & out[17]);
    bc[1] = out[16] ^ ((~out[17]) & out[18]);
    bc[2] = out[17] ^ ((~out[18]) & out[19]);
    bc[3] = out[18] ^ ((~out[19]) & out[15]);
    bc[4] = out[19] ^ ((~out[15]) & out[16]);
    out[15] = bc[0];
    out[16] = bc[1];
    out[17] = bc[2];
    out[18] = bc[3];
    out[19] = bc[4];

    bc[0] = out[20] ^ ((~out[21]) & out[22]);
    bc[1] = out[21] ^ ((~out[22]) & out[23]);
    bc[2] = out[22] ^ ((~out[23]) & out[24]);
    bc[3] = out[23] ^ ((~out[24]) & out[20]);
    bc[4] = out[24] ^ ((~out[20]) & out[21]);
    out[20] = bc[0];
    out[21] = bc[1];
    out[22] = bc[2];
    out[23] = bc[3];
    out[24] = bc[4];
}


__global__
void kernel_keccak_f800_lm8()
{
    uint32_t flat_matrix[25];

    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x00000001u;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x00008082u;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x0000808au;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x80008000u;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x0000808bu;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x80000001u;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x80008081u;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x00008009u;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x0000008au;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x00000088u;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x80008009u;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x8000000au;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x8000808bu;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x0000008bu;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x00008089u;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x00008003u;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x00008002u;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x00000080u;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x0000800au;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x8000000au;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x80008081u;
    keccak_f800_round_lm8(flat_matrix); flat_matrix[0] ^= 0x00008080u;
}


__host__
bool keccak_f800_lm8(
    cudaStream_t stream,
    uint32_t const blocks,
    uint32_t const threads)
{
    kernel_keccak_f800_lm8<<<blocks, threads, 0, stream>>>();
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
