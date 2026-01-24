#pragma once


__device__ __forceinline__
uint2 operator^(
    uint2 const a,
    uint32_t const  b)
{
    return make_uint2(a.x ^ b, a.y);
}


__device__ __forceinline__
uint2 operator^(
    uint2 const a,
    uint2 const b)
{
    return make_uint2(a.x ^ b.x, a.y ^ b.y);
}


__device__ __forceinline__
uint2 operator~(uint2 const a)
{
    return make_uint2(~a.x, ~a.y);
}


__device__ __forceinline__
void operator^=(
    uint2& a,
    uint2 const b)
{
    asm
    (
        "xor.b32 %0, %0, %2;\n"
        "xor.b32 %1, %1, %3;\n"
        : "+r"(a.x),
          "+r"(a.y)
        : "r"(b.x),
          "r"(b.y)
    ); //  a = a ^ b;
}
