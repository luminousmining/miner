#pragma once


__device__ __forceinline__
uint32_t fnv1(
    uint32_t const u,
    uint32_t const v)
{
    return (u * 0x01000193u) ^ v;
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


#define FNV1_OFFSET 0x811c9dc5
#define FNV1_PRIME 0x01000193u


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
    return (u ^ v) * FNV1_PRIME;
}
