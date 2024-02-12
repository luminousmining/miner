#pragma once


__device__ __forceinline__
uint64_t xor5(
    uint64_t const* const arr,
    uint32_t const i)
{
    return arr[i]
         ^ arr[i + 5u]
         ^ arr[i + 10u]
         ^ arr[i + 15u]
         ^ arr[i + 20u];
}


__device__ __forceinline__
uint64_t xor5(
    uint32_t const* const arr,
    uint32_t const i)
{
    return arr[i]
         ^ arr[i + 5u]
         ^ arr[i + 10u]
         ^ arr[i + 15u]
         ^ arr[i + 20u];
}
