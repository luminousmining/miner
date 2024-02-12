#include <algo/crypto/fnv1.hpp>
#include <algo/crypto/kiss99.hpp>
#include <algo/progpow/progpow.hpp>
#include <common/custom.hpp>



algo::Kiss99Properties algo::progpow::initializeRound(
    uint64_t const period,
    int32_t* const dst,
    int32_t* const src)
{
    ////////////////////////////////////////////////////////////////////////////
    algo::Kiss99Properties data{};
    data.z = algo::fnv1a(algo::FNV1_OFFSET, period & 0xffffffff);
    data.w = algo::fnv1a(data.z, period >> 32);
    data.jsr = algo::fnv1a(data.w, period);
    data.jcong = algo::fnv1a(data.jsr, period >> 32);

    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i{ 0u }; i < algo::progpow::REGS; ++i)
    {
        dst[i] = i;
    }
    for (uint32_t i{ 0u }; i < algo::progpow::REGS; ++i)
    {
        src[i] = i;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint32_t indexDst{ 0u };
    uint32_t indexSrc{ 0u };
    for (uint32_t i{ algo::progpow::REGS - 1u }; i > 0u; --i)
    {
        indexDst = algo::kiss99(data) % (i + 1u);
        indexSrc = algo::kiss99(data) % (i + 1u);

        common::swap(&dst[i], &dst[indexDst]);
        common::swap(&src[i], &src[indexSrc]);
    }

    ////////////////////////////////////////////////////////////////////////////
    return data;
}
