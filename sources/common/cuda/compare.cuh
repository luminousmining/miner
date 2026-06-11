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


// Little-endian variant: the most-significant byte lives at the highest index,
// so the comparison scans from N-1 down to 0. Returns true when r <= l.
template<typename T, uint32_t N>
__forceinline__ __device__
bool isLowerLittleEndian(
    T const* __restrict__ r,
    T const* __restrict__ l)
{
    #pragma unroll
    for (int32_t i{ static_cast<int32_t>(N) - 1 }; 0 <= i; --i)
    {
        if (r[i] < l[i])
        {
            return true;
        }
        if (l[i] < r[i])
        {
            return false;
        }
    }

    return true;
}
