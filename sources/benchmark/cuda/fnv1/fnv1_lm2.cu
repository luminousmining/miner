///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
#include <common/error/cuda_error.hpp>

///////////////////////////////////////////////////////////////////////////////
constexpr uint32_t FNV1_PRIME{ 0x01000193u };


__global__
void kernel_fnv1_lm2(
    uint32_t* __restrict__ const output,
    uint32_t const u,
    uint32_t const v)
{
    uint32_t const thread_id{ (blockIdx.x * blockDim.x) + threadIdx.x };
    uint32_t const u2{ u + thread_id };
    uint32_t const v2{ v + thread_id };
    uint32_t const result{ __umulhi(u2, FNV1_PRIME) ^ v2 };
    output[thread_id] = result;
}


__host__
bool fnv1_lm2(
    cudaStream_t stream,
    uint32_t* const result,
    uint32_t const blocks,
    uint32_t const threads)
{
    kernel_fnv1_lm2<<<blocks, threads, 0, stream>>>
    (
        result,
        0u,
        0u
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
