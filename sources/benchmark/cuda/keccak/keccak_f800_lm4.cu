///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
#include <common/cuda/rotate_byte.cuh>
#include <common/cuda/xor.cuh>

///////////////////////////////////////////////////////////////////////////////
#include <common/error/cuda_error.hpp>


__device__ __forceinline__
void keccak_f800_round_lm4(
    uint32_t* const __restrict__ out)
{
    uint32_t tmp;
    uint32_t bc[5];

    // Theta
    bc[0] = xor5(out, 0u);
    bc[1] = xor5(out, 1u);
    bc[2] = xor5(out, 2u);
    bc[3] = xor5(out, 3u);
    bc[4] = xor5(out, 4u);

    uint32_t const d0 = bc[4] ^ rol_u32(bc[1], 1u);
    uint32_t const d1 = bc[0] ^ rol_u32(bc[2], 1u);
    uint32_t const d2 = bc[1] ^ rol_u32(bc[3], 1u);
    uint32_t const d3 = bc[2] ^ rol_u32(bc[4], 1u);
    uint32_t const d4 = bc[3] ^ rol_u32(bc[0], 1u);

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
    tmp = out[1];
    out[1]  = rol_u32(out[6], 12u);
    out[6]  = rol_u32(out[9], 20u);
    out[9]  = rol_u32(out[22], 29u);
    out[22] = rol_u32(out[14], 7u);
    out[14] = rol_u32(out[20], 18u);
    out[20] = rol_u32(out[2], 30u);
    out[2]  = rol_u32(out[12], 11u);
    out[12] = rol_u32(out[13], 25u);
    out[13] = rol_u32(out[19], 8u);
    out[19] = rol_u32(out[23], 24u);
    out[23] = rol_u32(out[15], 9u);
    out[15] = rol_u32(out[4], 27u);
    out[4]  = rol_u32(out[24], 14u);
    out[24] = rol_u32(out[21], 2u);
    out[21] = rol_u32(out[8], 23u);
    out[8]  = rol_u32(out[16], 13u);
    out[16] = rol_u32(out[5], 4u);
    out[5]  = rol_u32(out[3], 28u);
    out[3]  = rol_u32(out[18], 21u);
    out[18] = rol_u32(out[17], 15u);
    out[17] = rol_u32(out[11], 10u);
    out[11] = rol_u32(out[7], 6u);
    out[7]  = rol_u32(out[10], 3u);
    out[10] = rol_u32(tmp, 1u);

    // Chi
    #pragma unroll
    for (uint32_t i{ 0u }; i < 5u; ++i)
    {
        tmp = i * 5u;
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


__global__
void kernel_keccak_f800_lm4()
{
    uint32_t flat_matrix[25];

    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x00000001u;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x00008082u;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x0000808au;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x80008000u;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x0000808bu;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x80000001u;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x80008081u;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x00008009u;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x0000008au;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x00000088u;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x80008009u;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x8000000au;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x8000808bu;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x0000008bu;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x00008089u;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x00008003u;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x00008002u;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x00000080u;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x0000800au;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x8000000au;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x80008081u;
    keccak_f800_round_lm4(flat_matrix); flat_matrix[0] ^= 0x00008080u;
}


__host__
bool keccak_f800_lm4(
    cudaStream_t stream,
    uint32_t const blocks,
    uint32_t const threads)
{
    kernel_keccak_f800_lm4<<<blocks, threads, 0, stream>>>();
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
