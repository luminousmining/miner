#pragma once


__device__ __forceinline__
uint64_t be_u64(
    uint64_t const value)
{
    uint2 t;
    asm("mov.b64 {%0,%1},%2; \n\t" : "=r"(t.x), "=r"(t.y) : "l"(value));
    t.x = __byte_perm(t.x, 0, 0x0123);
    t.y = __byte_perm(t.y, 0, 0x0123);

    uint64_t result;
    asm("mov.b64 %0,{%1,%2}; \n\t" : "=l"(result) : "r"(t.y), "r"(t.x));
    return result;
}
