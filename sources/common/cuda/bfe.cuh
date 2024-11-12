#pragma once


__device__ __forceinline__
uint32_t bfe(
    uint32_t const x,
    uint32_t const bit,
    uint32_t const numBits)
{
    uint32_t ret;
    asm volatile(
        "bfe.u32 %0, %1, %2, %3;"
        : "=r"(ret) : "r"(x), "r"(bit), "r"(numBits));
    return ret;
}
