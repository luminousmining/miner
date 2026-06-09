// Blake3 (Alephium) OpenCL kernels. BIT-IDENTICAL to sources/algo/blake3/blake3_pow.cpp.
// PoW = BLAKE3(BLAKE3(nonce(24) || headerBlob(302))) over 326 bytes. The 24-byte nonce is
// prepended: big-endian 8-byte search value + 16 zero bytes (== the 48-hex submit string);
// headerBlob is left-aligned in the header buffer (words 0..75).

#ifndef MAX_RESULT
#define MAX_RESULT 4
#endif

#define IV0 0x6A09E667u
#define IV1 0xBB67AE85u
#define IV2 0x3C6EF372u
#define IV3 0xA54FF53Au
#define IV4 0x510E527Fu
#define IV5 0x9B05688Cu
#define IV6 0x1F83D9ABu
#define IV7 0x5BE0CD19u

__constant uchar SCHEDULE[7][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8 },
    { 3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1 },
    { 10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6 },
    { 12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4 },
    { 9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7 },
    { 11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13 },
};

inline uint ror32(uint const x, uint const n)
{
    return (x >> n) | (x << (32u - n));
}

inline uint bswap32(uint const x)
{
    return (x >> 24) | ((x >> 8) & 0x0000FF00u) | ((x << 8) & 0x00FF0000u) | (x << 24);
}

#define G(st, a, b, c, d, x, y)                 \
    {                                           \
        st[a] = st[a] + st[b] + x;              \
        st[d] = ror32(st[d] ^ st[a], 16u);      \
        st[c] = st[c] + st[d];                  \
        st[b] = ror32(st[b] ^ st[c], 12u);      \
        st[a] = st[a] + st[b] + y;              \
        st[d] = ror32(st[d] ^ st[a], 8u);       \
        st[c] = st[c] + st[d];                  \
        st[b] = ror32(st[b] ^ st[c], 7u);       \
    }

void compress(uint* vector, uint const* block, uint const blockLen, uint const flags)
{
    uint st[16];
    st[0] = vector[0]; st[1] = vector[1]; st[2] = vector[2]; st[3] = vector[3];
    st[4] = vector[4]; st[5] = vector[5]; st[6] = vector[6]; st[7] = vector[7];
    st[8] = IV0; st[9] = IV1; st[10] = IV2; st[11] = IV3;
    st[12] = 0u; st[13] = 0u; st[14] = blockLen; st[15] = flags;

    for (uint r = 0u; r < 7u; ++r)
    {
        __constant uchar* s = SCHEDULE[r];
        G(st, 0u, 4u, 8u,  12u, block[s[0]],  block[s[1]]);
        G(st, 1u, 5u, 9u,  13u, block[s[2]],  block[s[3]]);
        G(st, 2u, 6u, 10u, 14u, block[s[4]],  block[s[5]]);
        G(st, 3u, 7u, 11u, 15u, block[s[6]],  block[s[7]]);
        G(st, 0u, 5u, 10u, 15u, block[s[8]],  block[s[9]]);
        G(st, 1u, 6u, 11u, 12u, block[s[10]], block[s[11]]);
        G(st, 2u, 7u, 8u,  13u, block[s[12]], block[s[13]]);
        G(st, 3u, 4u, 9u,  14u, block[s[14]], block[s[15]]);
    }

    for (uint i = 0u; i < 8u; ++i)
    {
        vector[i] = st[i] ^ st[i + 8u];
    }
}

void initVector(uint* v)
{
    v[0] = IV0; v[1] = IV1; v[2] = IV2; v[3] = IV3;
    v[4] = IV4; v[5] = IV5; v[6] = IV6; v[7] = IV7;
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
    compress(vector, &buf[0], 64u, 1u);                 // CHUNK_START
    for (uint i = 0u; i < 4u; ++i)
    {
        compress(vector, &buf[16u * (i + 1u)], 64u, 0u);// EMPTY
    }
    uint last[16];
    for (uint i = 0u; i < 16u; ++i) { last[i] = 0u; }
    last[0] = buf[80];
    last[1] = buf[81] & 0x0000FFFFu;
    compress(vector, last, 6u, 10u);                    // CHUNK_END | ROOT

    uint block2[16];
    for (uint i = 0u; i < 16u; ++i) { block2[i] = 0u; }
    for (uint i = 0u; i < 8u; ++i)  { block2[i] = vector[i]; }
    initVector(vector);
    compress(vector, block2, 32u, 11u);                 // CHUNK_START|END|ROOT

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
