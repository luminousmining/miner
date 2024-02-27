#pragma once


__device__ __forceinline__
uint32_t be_u32(
    uint32_t const value)
{
    return __byte_perm(value, value, 0x0123);
}

__device__ __forceinline__
uint32_t be_u32(
    uint32_t const x,
    uint32_t const y)
{
    return __byte_perm(x, y, 0x0123);
}
