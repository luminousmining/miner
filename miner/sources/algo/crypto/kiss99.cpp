#include <algo/crypto/kiss99.hpp>


uint32_t algo::kiss99(
    algo::Kiss99Properties& data)
{
    data.z = (36969u * (data.z & 0xffff)) + (data.z >> 16);
    data.w = (18000u * (data.w & 0xffff)) + (data.w >> 16);

    uint32_t mwc{ (data.z << 16) + data.w };

    data.jsr ^= (data.jsr << 17);
    data.jsr ^= (data.jsr >> 13);
    data.jsr ^= (data.jsr << 5);

    data.jcong = (69069u * data.jcong) + 1234567u;

    return (mwc ^ data.jcong) + data.jsr;
}
