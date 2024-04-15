#pragma once


template<typename T>
__forceinline__ __device__
bool isLowerOrEqual(
    T const r,
    T const l,
    uint32_t const length)
{
    #pragma unroll
    for (uint32_t i{ 0u }; i < length; ++i)
    {
        if (r[i] > l[i])
        {
            return false;
        }
        else if (r[i] < l[i])
        {
            return true;
        }
    }

    return true;
}
