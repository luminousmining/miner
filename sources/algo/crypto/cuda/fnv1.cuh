#pragma once

constexpr uint32_t FNV1_OFFSET{ 0x811c9dc5u };
constexpr uint32_t FNV1_PRIME{ 0x01000193u };


__device__ __forceinline__
uint32_t fnv1(
    uint32_t const u,
    uint32_t const v)
{
    uint32_t result;
    // (u * FNV1_PRIME) ^ v
    asm volatile
    (
        "mul.lo.u32 %0, %1, %2;\n"
        "xor.b32 %0, %0, %3;"
        : "=r"(result)     // %0
        : "r"(u),          // %1
          "r"(FNV1_PRIME), // %2
          "r"(v)           // %3
    );
    return result;
}


__device__ __forceinline__
void fnv1(
    uint4* __restrict__ v,
    uint4 const* __restrict__ const v2)
{
    v->x = fnv1(v->x, v2->x);
    v->y = fnv1(v->y, v2->y);
    v->z = fnv1(v->z, v2->z);
    v->w = fnv1(v->w, v2->w);
}


__device__ __forceinline__
void fnv1(
    uint4& v,
    uint4 const& v2)
{
    v.x = fnv1(v.x, v2.x);
    v.y = fnv1(v.y, v2.y);
    v.z = fnv1(v.z, v2.z);
    v.w = fnv1(v.w, v2.w);
}


__device__ __forceinline__
uint32_t fnv1_reduce(
    uint4 const& v)
{
    return fnv1(fnv1(fnv1(v.x, v.y), v.z), v.w);
}


__device__ __forceinline__
uint32_t fnv1a(
    uint32_t const u,
    uint32_t const v)
{
    uint32_t result;
    // (u ^ v) * FNV1_PRIME
    asm volatile
    (
        "xor.b32 %0, %1, %2;\n"
        "mul.lo.u32 %0, %0, %3;"
        : "=r"(result)    // %0
        : "r"(u),         // %1
          "r"(v),         // %2
          "r"(FNV1_PRIME) // %3
    );
    return result;
}
