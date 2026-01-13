#include <algo/fast_mod.hpp>


FastDivisor initFastMod(uint32_t const d)
{
    FastDivisor out{};

    out.divisor = d;

    if (0 == d || 1 == d)
    {
        out.magic = 0;
        out.shift = 0;
        return out;
    }

    uint32_t p{ 31u };
    uint64_t const one{ 1ull };

    while ((one << p) < d)
    {
        ++p;
    }

    uint64_t const m{ ((one << (p + 32u)) + d - 1ull) / d };

    out.magic = static_cast<uint32_t>(m);
    out.shift = p;

    return out;
}


uint32_t fastMod(FastDivisor const& divisor, uint32_t const value)
{
    uint32_t const d{ divisor.divisor };
    uint32_t const m{ divisor.magic };
    uint32_t const s{ divisor.shift };

    uint64_t const mul{ static_cast<uint64_t>(value) * static_cast<uint64_t>(m) };
    uint32_t q{ static_cast<uint32_t>(mul >> (32u + s)) };

    uint32_t r{ value - (q * d) };

    while (r >= d)
    {
        r -= d;
    }

    return r;
}
