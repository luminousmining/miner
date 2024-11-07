#pragma once

__device__ __forceinline__
void toU64(
    uint64_t* const ptr,
    uint32_t const index,
    uint4 const& v)
{
    asm volatile(
        "mov.b64 %0,{%1,%2}; \n\t"
        : "=l"(ptr[index])
        : "r"(v.x), "r"(v.y));
    asm volatile(
        "mov.b64 %0,{%1,%2}; \n\t"
        : "=l"(ptr[index + 1])
        : "r"(v.z), "r"(v.w));
}
