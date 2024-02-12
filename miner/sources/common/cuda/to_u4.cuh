#pragma once


__device__ __forceinline__
uint4 toU4(
    uint64_t const a,
    uint64_t const b)
{
    uint4 result;
    asm("mov.b64 {%0,%1},%2; \n\t" : "=r"(result.x), "=r"(result.y) : "l"(a));
    asm("mov.b64 {%0,%1},%2; \n\t" : "=r"(result.z), "=r"(result.w) : "l"(b));
    return result;
}
