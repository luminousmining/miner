#pragma once


__device__ __forceinline__
void copy_u4(
    uint4& dst,
    uint4 const& src)
{
    dst.x = src.x;
    dst.y = src.y;
    dst.z = src.z;
    dst.w = src.w;
}
