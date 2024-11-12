#pragma once


__device__ __forceinline__
uint2 vectorize(uint64_t const x)
{
    uint2 result;
    asm volatile(
        "mov.b64 {%0,%1},%2; \n\t"
        : "=r"(result.x), "=r"(result.y)
        : "l"(x));
    return result;
}


__device__ __forceinline__
uint4 vectorize_u2(
    uint2 const x,
    uint2 const y)
{
    uint4 result;

    result.x = x.x;
    result.y = x.y;
    result.z = y.x;
    result.w = y.y;

    return result;
}


template<typename TVector>
__device__ __forceinline__
uint64_t devectorize(TVector const vector)
{
    uint64_t result;
    asm volatile(
        "mov.b64 %0,{%1,%2}; \n\t"
        : "=l"(result)
        : "r"(vector.x), "r"(vector.y));
    return result;
}


template<typename TVector3, typename TVector2>
__device__ __forceinline__
void devectorize(
    TVector3 const from,
    TVector2& dest1,
    TVector2& dest2)
{
    dest1.x = from.x;
    dest1.y = from.y;
    dest2.x = from.z;
    dest2.y = from.w;
}
