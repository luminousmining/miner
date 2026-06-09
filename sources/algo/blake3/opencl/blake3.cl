// Blake3 (Alephium) OpenCL mining kernels. The BLAKE3 compression primitive is the
// shared crypto kernel (sources/algo/crypto/opencl/blake3.cl); only the Alephium
// mining orchestration (double-BLAKE3 over the 326-byte input) lives here.
// PoW = BLAKE3(BLAKE3(nonce(24) || headerBlob(302))) over 326 bytes. The 24-byte nonce is
// prepended: big-endian 8-byte search value + 16 zero bytes (== the 48-hex submit string);
// headerBlob is left-aligned in the header buffer (words 0..75).
//
// NOTE: the miner's kernel generator resolves this include; the host KAT
// (opencl/tests/blake3_kat.cpp) concatenates the primitive source instead.
#include "kernel/crypto/blake3.cl"

#ifndef MAX_RESULT
#define MAX_RESULT 4
#endif

inline uint bswap32(uint const x)
{
    return (x >> 24) | ((x >> 8) & 0x0000FF00u) | ((x << 8) & 0x00FF0000u) | (x << 24);
}

// Thin adapter over the shared full-XOF blake3_compress: keep only the 8-word chaining
// value, matching the Alephium single-chunk orchestration below (counter always 0).
// Pointers are explicitly __private: blake3_compress's array parameters decay to
// const __private uint*, and the ROCm/comgr compiler (unlike AMD-APP) rejects passing
// generic-address-space pointers to them. All call sites below pass private locals.
void compress8(__private uint* vector, __private uint const* block, uint const blockLen, uint const flags)
{
    uint out16[16];
    blake3_compress(vector, block, 0u, 0u, blockLen, flags, out16);
    for (uint i = 0u; i < 8u; ++i)
    {
        vector[i] = out16[i];
    }
}

void initVector(uint* v)
{
    v[0] = B3_IV0; v[1] = B3_IV1; v[2] = B3_IV2; v[3] = B3_IV3;
    v[4] = B3_IV4; v[5] = B3_IV5; v[6] = B3_IV6; v[7] = B3_IV7;
}

// Alephium mining input (326 B) = nonce(24) || headerBlob(302). nonce = big-endian
// 8-byte search value + 16 zero bytes; header holds headerBlob left-aligned (words 0..75,
// read straight from global). out8 = 8 u32 digest = BLAKE3(BLAKE3(mining input)).
// Only buf[0..81] (82 words) are hashed; kept small to ease register pressure/occupancy.
void powHash(__global uint const* header, ulong const nonce, uint* out8)
{
    uint buf[82];
    for (uint i = 0u; i < 82u; ++i)
    {
        buf[i] = 0u;
    }
    buf[0] = bswap32((uint)(nonce >> 32));
    buf[1] = bswap32((uint)(nonce & 0xFFFFFFFFul));
    // buf[2..5] = 0  (nonce bytes 8..23)
    for (uint i = 0u; i < 76u; ++i)
    {
        buf[6u + i] = header[i];   // headerBlob shifted up by 24 bytes (6 words)
    }

    uint vector[8];
    initVector(vector);
    compress8(vector, &buf[0], 64u, 1u);                 // CHUNK_START
    for (uint i = 0u; i < 4u; ++i)
    {
        compress8(vector, &buf[16u * (i + 1u)], 64u, 0u);// EMPTY
    }
    uint last[16];
    for (uint i = 0u; i < 16u; ++i) { last[i] = 0u; }
    last[0] = buf[80];
    last[1] = buf[81] & 0x0000FFFFu;
    compress8(vector, last, 6u, 10u);                    // CHUNK_END | ROOT

    uint block2[16];
    for (uint i = 0u; i < 16u; ++i) { block2[i] = 0u; }
    for (uint i = 0u; i < 8u; ++i)  { block2[i] = vector[i]; }
    initVector(vector);
    compress8(vector, block2, 32u, 11u);                 // CHUNK_START|END|ROOT

    for (uint i = 0u; i < 8u; ++i)
    {
        out8[i] = vector[i];
    }
}

// digest <= target, byte-wise from index 0 (matches common/cuda/compare.cuh).
inline bool isLowerOrEqual(uchar const* r, uchar const* l)
{
    for (uint i = 0u; i < 32u; ++i)
    {
        if (r[i] > l[i]) { return false; }
        if (r[i] < l[i]) { return true; }
    }
    return true;
}

typedef struct __attribute__((aligned(8)))
{
    uchar found;
    uint  count;
    ulong nonces[MAX_RESULT];
} Result;

// KAT entry: digest for one nonce. header = headerBlob words (left-aligned), out = 8 u32.
__kernel void test_hash(__global uint const* header, ulong const nonce, __global uint* out)
{
    uint h[8];
    powHash(header, nonce, h);
    for (uint i = 0u; i < 8u; ++i) { out[i] = h[i]; }
}

// Mining kernel: nonce = startNonce + gid. Hit iff digest <= target AND the
// chain index (digest byte 31 % 16) maps to (fromGroup, toGroup).
__kernel void search(__global uint const*  header,
                     __global uchar const* target,
                     ulong const           startNonce,
                     uint const            fromGroup,
                     uint const            toGroup,
                     __global Result*      result)
{
    ulong const nonce = startNonce + (ulong)get_global_id(0);

    uint h[8];
    powHash(header, nonce, h);

    uchar tgt[32];
    for (uint i = 0u; i < 32u; ++i) { tgt[i] = target[i]; }

    if (isLowerOrEqual((uchar const*)h, tgt))
    {
        uint const bigIndex = (h[7] >> 24) % 16u;
        if ((bigIndex / 4u) == fromGroup && (bigIndex % 4u) == toGroup)
        {
            uint const idx = atomic_inc(&result->count);
            result->found = 1;
            if (idx < MAX_RESULT)
            {
                result->nonces[idx] = nonce;
            }
        }
    }
}
