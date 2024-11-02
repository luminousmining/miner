#include <benchmark/cuda/kernels.hpp>


__global__
void kernel_ethash_v0()
{
}


__host__
bool ethash_v0(
        cudaStream_t stream,
        uint32_t const blocks,
        uint32_t const threads)
{
    kernel_ethash_v0<<<blocks, threads, 0, stream>>>();
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
