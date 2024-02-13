#pragma once


__device__ __forceinline__
uint32_t be_u32(
    uint32_t const value)
{
    return __byte_perm(value, value, 0x0123);
}
