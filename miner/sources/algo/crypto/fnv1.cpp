#include <algo/crypto/fnv1.hpp>


uint32_t algo::fnv1(
    uint32_t const u,
    uint32_t const v)
{
    return (u * algo::FNV1_PRIME) ^ v;
}


uint32_t algo::fnv1a(
    uint32_t const u,
    uint32_t const v)
{
    return  (u ^ v) * algo::FNV1_PRIME;
}
