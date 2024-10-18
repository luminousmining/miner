#pragma once


__device__ __forceinline__
unsigned get_lane_id()
{
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}
