#include <algo/math.hpp>


bool algo::isOddPrime(
    uint64_t const number)
{
    for (uint64_t d{ 3u }; (d * d) <= number; d += 2)
    {
        if (0u == (number % d))
        {
            return false;
        }
    }

    return true;
}


uint64_t algo::largestPrime(
    uint64_t number)
{
    if (2 > number)
    {
        return 0;
    }

    if (2 == number)
    {
        return 2;
    }

    if (0 == (number % 2))
    {
        --number;
    }

    while (false == algo::isOddPrime(number))
    {
        number -= 2;
    }

    return number;
}
