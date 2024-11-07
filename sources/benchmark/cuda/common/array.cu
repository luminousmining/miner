#include <benchmark/cuda/kernels.hpp>


__global__
void kernel_init_array(
    uint32_t* const dest,
    uint64_t const size)
{
    for (uint64_t i = 0ull; i < size; ++i)
    {
        dest[i] = i;
    }
}


__host__
bool init_array(
        cudaStream_t stream,
        uint32_t* const dest,
        uint64_t const size)
{
    kernel_init_array<<<1, 1, 0, stream>>>(dest, size);
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
