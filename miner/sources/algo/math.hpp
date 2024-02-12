#pragma once

#include <cstdint>
#include <cstddef>

namespace algo
{
    bool isOddPrime(uint64_t number);
    uint64_t primeLarge(uint64_t number);

    template<typename T>
    T min(T const l, T const r)
    {
        if (l < r)
        {
            return l;
        }
        return r;
    }

    template<typename T>
    T max(T const l, T const r)
    {
        if (l > r)
        {
            return l;
        }
        return r;
    }

    template<typename T>
    bool inRange(T const& min, T const& max, T const& value)
    {
        if (value >= min && value <= max)
        {
            return true;
        }
        return false;
    }
}
