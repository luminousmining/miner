__device__ __forceinline__
uint2 vectorize(uint64_t const x)
{
    uint2 result;
    asm("mov.b64 {%0,%1},%2; \n\t" : "=r"(result.x), "=r"(result.y) : "l"(x));
    return result;
}


__device__ __forceinline__
uint64_t devectorize(uint2 const x)
{
    uint64_t result;
    asm("mov.b64 %0,{%1,%2}; \n\t" : "=l"(result) : "r"(x.x), "r"(x.y));
    return result;
}
