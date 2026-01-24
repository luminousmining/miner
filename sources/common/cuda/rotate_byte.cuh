#pragma once

#if !defined(__KERNEL_COMPILED)
#include <common/cuda/vectorize.cuh>
#else
#include "kernel/common/vectorize.cuh"
#endif


__device__ __forceinline__
uint64_t rol_u64(
    uint64_t const value,
    uint32_t const offset)
{
    return (value << offset) | (value >> (64u - offset));
}


__device__ __forceinline__
uint64_t rol_u32(
    uint32_t const value,
    uint32_t const offset)
{
    return __funnelshift_l(value, value, offset);
}


__device__ __forceinline__
uint32_t ror_u32(
    uint32_t const x,
    uint32_t const n)
{
    return __funnelshift_r(x, x, n);
}


// __device__ __forceinline__
// uint64_t ror_64(
//     uint64_t const b,
//     uint32_t const offset)
// {
//     uint2 a;
//     uint2 result;
//     a = vectorize(b);

//     if (offset < 32u)
//     {
//         asm volatile
//         (
//             "shf.r.wrap.b32 %0, %1, %2, %3;"
//             : "=r"(result.x)
//             : "r"(a.x),
//               "r"(a.y),
//               "r"(offset)
//         );
//         asm volatile
//         (
//             "shf.r.wrap.b32 %0, %1, %2, %3;"
//             : "=r"(result.y)
//             : "r"(a.y),
//               "r"(a.x),
//               "r"(offset)
//         );
//     }
//     else
//     {
//         asm volatile
//         (
//             "shf.r.wrap.b32 %0, %1, %2, %3;"
//             : "=r"(result.x)
//             : "r"(a.y),
//               "r"(a.x),
//               "r"(offset)
//         );
//         asm volatile
//         (
//             "shf.r.wrap.b32 %0, %1, %2, %3;"
//             : "=r"(result.y)
//             : "r"(a.x),
//               "r"(a.y),
//               "r"(offset)
//         );
//     }

//     return devectorize(result);
// }


__device__ __forceinline__
uint64_t ror_64(uint64_t const b, uint32_t const offset)
{
    uint2 a = vectorize(b);
    uint2 result;
    asm("{\n\t"
        ".reg .b64 tmp, input;\n"
        "mov.b64 input, {%2,%3};\n"
        "shr.b64 tmp, input, %4;\n"
        "shl.b64 input, input, %5;\n"
        "or.b64 tmp, input, tmp;\n"
        "mov.b64 {%0,%1}, tmp;\n"
        "}"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.x), "r"(a.y), "r"(offset & 63u), "r"((64u - offset) & 63u));
    return devectorize(result);
}
