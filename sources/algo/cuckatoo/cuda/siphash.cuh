#pragma once

////////////////////////////////////////////////////////////////////////////
// SipHash-2-4 device functions for Cuckatoo32 edge generation.
//
// Keys k0..k3 come from:
//   H1 = blake2b-256(pre_pow || nonce_le8)
//   H2 = blake2b-256(H1)
//   k0 = le64(H2[ 0.. 7])
//   k1 = le64(H2[ 8..15])
//   k2 = le64(H2[16..23])
//   k3 = le64(H2[24..31])
//
// Edge generation:
//   uNode(e) = (uint32_t) sipHash24(k, 2*e  )   (implicit & 0xFFFFFFFF)
//   vNode(e) = (uint32_t) sipHash24(k, 2*e+1)
//
// Reference: https://github.com/tromp/cuckoo/blob/master/src/siphash.hpp
////////////////////////////////////////////////////////////////////////////

#if defined(CUDA_ENABLE)

#include <cuda_runtime.h>
#include <stdint.h>


__device__ __forceinline__
uint64_t sip_rotl64(uint64_t const x, int const n)
{
    return (x << n) | (x >> (64 - n));
}


#define SIPROUND(v0, v1, v2, v3)                              \
    do {                                                      \
        (v0) += (v1);                                         \
        (v1)  = sip_rotl64((v1), 13) ^ (v0);                 \
        (v0)  = sip_rotl64((v0), 32);                         \
        (v2) += (v3);                                         \
        (v3)  = sip_rotl64((v3), 16) ^ (v2);                 \
        (v0) += (v3);                                         \
        (v3)  = sip_rotl64((v3), 21) ^ (v0);                 \
        (v2) += (v1);                                         \
        (v1)  = sip_rotl64((v1), 17) ^ (v2);                 \
        (v2)  = sip_rotl64((v2), 32);                         \
    } while (0)


////////////////////////////////////////////////////////////////////////////
/// SipHash-2-4 as used by Cuckoo Cycle.
/// Keys are the 4 × uint64 words from the 32-byte blake2b seed.
////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
uint64_t sipHash24(
    uint64_t const k0, uint64_t const k1,
    uint64_t const k2, uint64_t const k3,
    uint64_t const nonce)
{
    uint64_t v0 = k0;
    uint64_t v1 = k1;
    uint64_t v2 = k2;
    uint64_t v3 = k3 ^ nonce;

    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);

    v0 ^= nonce;
    v2 ^= 0xffull;

    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);
    SIPROUND(v0, v1, v2, v3);

    return v0 ^ v1 ^ v2 ^ v3;
}

#undef SIPROUND


////////////////////////////////////////////////////////////////////////////
/// U-node (partition 0) for edge index @p edge.
////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
uint32_t sipNodeU(
    uint64_t const k0, uint64_t const k1,
    uint64_t const k2, uint64_t const k3,
    uint64_t const edge)
{
    return static_cast<uint32_t>(sipHash24(k0, k1, k2, k3, edge * 2ULL));
}


////////////////////////////////////////////////////////////////////////////
/// V-node (partition 1) for edge index @p edge.
////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__
uint32_t sipNodeV(
    uint64_t const k0, uint64_t const k1,
    uint64_t const k2, uint64_t const k3,
    uint64_t const edge)
{
    return static_cast<uint32_t>(sipHash24(k0, k1, k2, k3, edge * 2ULL + 1ULL));
}

#endif // CUDA_ENABLE
