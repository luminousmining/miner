#pragma once

#include <algo/kheavyhash/cuda/kheavyhash_device.cuh>
#include <algo/kheavyhash/result.hpp>
#include <resolver/nvidia/kheavyhash_kernel_parameter.hpp>


// Per-nonce mining kernel: each thread tries nonce = startNonce + global thread id,
// computes powHash -> heavyHash, and publishes a hit (pow <= target, little-endian)
// into the shared Result. The matrix is generated host-side and uploaded; the GPU
// never does matrix generation. The device hash functions are KAT-verified against
// the CPU reference (see tests/cuda_device_test.cpp).
__global__
void kernel_kheavyhash_search(
    algo::kheavyhash::Result* __restrict__ result,
    uint16_t const* __restrict__ matrix,
    uint8_t const* __restrict__ header,
    uint8_t const* __restrict__ target,
    uint64_t const timestamp,
    uint64_t const startNonce)
{
    uint64_t const nonce{ startNonce + (uint64_t)((blockIdx.x * blockDim.x) + threadIdx.x) };

    uint8_t pre[32];
    uint8_t tgt[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i)
    {
        pre[i] = header[i];
        tgt[i] = target[i];
    }

    uint8_t h1[32];
    kheavyhash_cuda::powHash(pre, timestamp, nonce, h1);
    uint8_t pow[32];
    kheavyhash_cuda::heavyHash(matrix, h1, pow);

    if (true == kheavyhash_cuda::meetsTarget(pow, tgt))
    {
        uint32_t const index{ atomicAdd((uint32_t*)&result->count, 1u) };
        result->found = true;
        if (index < algo::kheavyhash::MAX_RESULT)
        {
            result->nonces[index] = nonce;
        }
    }
}


__host__
void kheavyhashSearch(
    cudaStream_t stream,
    resolver::nvidia::kheavyhash::KernelParameters& params,
    uint32_t const currentIndexStream,
    uint32_t const blocks,
    uint32_t const threads)
{
    kernel_kheavyhash_search<<<blocks, threads, 0, stream>>>(
        &params.resultCache[currentIndexStream],
        params.matrix,
        params.header->ubytes,
        params.target->ubytes,
        params.hostTimestamp,
        params.hostNonce);
}
