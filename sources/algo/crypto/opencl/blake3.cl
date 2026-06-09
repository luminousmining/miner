// Single-chunk BLAKE3 for OpenCL (input <= 1024 bytes => one chunk, chunk counter 0).
// FishHash only needs blake3 over the 180-byte header (-> 64-byte seed) and over a
// 96-byte buffer (-> 32-byte digest), both single-chunk. Not a general streaming hash.
//
// Verified bit-for-bit against the vendored reference blake3 (see crypto/opencl/tests).

#ifndef LM_BLAKE3_CL
#define LM_BLAKE3_CL

#define B3_IV0 0x6A09E667u
#define B3_IV1 0xBB67AE85u
#define B3_IV2 0x3C6EF372u
#define B3_IV3 0xA54FF53Au
#define B3_IV4 0x510E527Fu
#define B3_IV5 0x9B05688Cu
#define B3_IV6 0x1F83D9ABu
#define B3_IV7 0x5BE0CD19u

#define B3_FLAG_CHUNK_START 1u
#define B3_FLAG_CHUNK_END 2u
#define B3_FLAG_ROOT 8u

// BLAKE3 message permutation schedule (round r, position i) -> message word index.
#define B3_Z00 0
#define B3_Z01 1
#define B3_Z02 2
#define B3_Z03 3
#define B3_Z04 4
#define B3_Z05 5
#define B3_Z06 6
#define B3_Z07 7
#define B3_Z08 8
#define B3_Z09 9
#define B3_Z0A 10
#define B3_Z0B 11
#define B3_Z0C 12
#define B3_Z0D 13
#define B3_Z0E 14
#define B3_Z0F 15
#define B3_Z10 2
#define B3_Z11 6
#define B3_Z12 3
#define B3_Z13 10
#define B3_Z14 7
#define B3_Z15 0
#define B3_Z16 4
#define B3_Z17 13
#define B3_Z18 1
#define B3_Z19 11
#define B3_Z1A 12
#define B3_Z1B 5
#define B3_Z1C 9
#define B3_Z1D 14
#define B3_Z1E 15
#define B3_Z1F 8
#define B3_Z20 3
#define B3_Z21 4
#define B3_Z22 10
#define B3_Z23 12
#define B3_Z24 13
#define B3_Z25 2
#define B3_Z26 7
#define B3_Z27 14
#define B3_Z28 6
#define B3_Z29 5
#define B3_Z2A 9
#define B3_Z2B 0
#define B3_Z2C 11
#define B3_Z2D 15
#define B3_Z2E 8
#define B3_Z2F 1
#define B3_Z30 10
#define B3_Z31 7
#define B3_Z32 12
#define B3_Z33 9
#define B3_Z34 14
#define B3_Z35 3
#define B3_Z36 13
#define B3_Z37 15
#define B3_Z38 4
#define B3_Z39 0
#define B3_Z3A 11
#define B3_Z3B 2
#define B3_Z3C 5
#define B3_Z3D 8
#define B3_Z3E 1
#define B3_Z3F 6
#define B3_Z40 12
#define B3_Z41 13
#define B3_Z42 9
#define B3_Z43 11
#define B3_Z44 15
#define B3_Z45 10
#define B3_Z46 14
#define B3_Z47 8
#define B3_Z48 7
#define B3_Z49 2
#define B3_Z4A 5
#define B3_Z4B 3
#define B3_Z4C 0
#define B3_Z4D 1
#define B3_Z4E 6
#define B3_Z4F 4
#define B3_Z50 9
#define B3_Z51 14
#define B3_Z52 11
#define B3_Z53 5
#define B3_Z54 8
#define B3_Z55 12
#define B3_Z56 15
#define B3_Z57 1
#define B3_Z58 13
#define B3_Z59 3
#define B3_Z5A 0
#define B3_Z5B 10
#define B3_Z5C 2
#define B3_Z5D 6
#define B3_Z5E 4
#define B3_Z5F 7
#define B3_Z60 11
#define B3_Z61 15
#define B3_Z62 5
#define B3_Z63 0
#define B3_Z64 1
#define B3_Z65 9
#define B3_Z66 8
#define B3_Z67 6
#define B3_Z68 14
#define B3_Z69 10
#define B3_Z6A 2
#define B3_Z6B 12
#define B3_Z6C 3
#define B3_Z6D 4
#define B3_Z6E 7
#define B3_Z6F 13

static inline uint b3_ror(uint x, uint n)
{
    return (x >> n) | (x << (32u - n));
}

#define B3_MX(r, i) (m[B3_Z##r##i])

#define B3_G(a, b, c, d, x, y)             \
    {                                      \
        s[a] = s[a] + s[b] + x;            \
        s[d] = b3_ror(s[d] ^ s[a], 16u);   \
        s[c] = s[c] + s[d];                \
        s[b] = b3_ror(s[b] ^ s[c], 12u);   \
        s[a] = s[a] + s[b] + y;            \
        s[d] = b3_ror(s[d] ^ s[a], 8u);    \
        s[c] = s[c] + s[d];                \
        s[b] = b3_ror(s[b] ^ s[c], 7u);    \
    }

#define B3_ROUND(r)                                          \
    {                                                        \
        B3_G(0x0, 0x4, 0x8, 0xC, B3_MX(r, 0), B3_MX(r, 1));  \
        B3_G(0x1, 0x5, 0x9, 0xD, B3_MX(r, 2), B3_MX(r, 3));  \
        B3_G(0x2, 0x6, 0xA, 0xE, B3_MX(r, 4), B3_MX(r, 5));  \
        B3_G(0x3, 0x7, 0xB, 0xF, B3_MX(r, 6), B3_MX(r, 7));  \
        B3_G(0x0, 0x5, 0xA, 0xF, B3_MX(r, 8), B3_MX(r, 9));  \
        B3_G(0x1, 0x6, 0xB, 0xC, B3_MX(r, A), B3_MX(r, B));  \
        B3_G(0x2, 0x7, 0x8, 0xD, B3_MX(r, C), B3_MX(r, D));  \
        B3_G(0x3, 0x4, 0x9, 0xE, B3_MX(r, E), B3_MX(r, F));  \
    }

// out[16] = full compression output (first 8 = chaining value, all 16 = XOF block 0).
static inline void blake3_compress(
    uint const cv[8],
    uint const m[16],
    uint const counter_lo,
    uint const counter_hi,
    uint const block_len,
    uint const flags,
    uint       out[16])
{
    uint s[16];
    s[0] = cv[0];
    s[1] = cv[1];
    s[2] = cv[2];
    s[3] = cv[3];
    s[4] = cv[4];
    s[5] = cv[5];
    s[6] = cv[6];
    s[7] = cv[7];
    s[8] = B3_IV0;
    s[9] = B3_IV1;
    s[10] = B3_IV2;
    s[11] = B3_IV3;
    s[12] = counter_lo;
    s[13] = counter_hi;
    s[14] = block_len;
    s[15] = flags;

    B3_ROUND(0);
    B3_ROUND(1);
    B3_ROUND(2);
    B3_ROUND(3);
    B3_ROUND(4);
    B3_ROUND(5);
    B3_ROUND(6);

    out[0] = s[0] ^ s[8];
    out[1] = s[1] ^ s[9];
    out[2] = s[2] ^ s[10];
    out[3] = s[3] ^ s[11];
    out[4] = s[4] ^ s[12];
    out[5] = s[5] ^ s[13];
    out[6] = s[6] ^ s[14];
    out[7] = s[7] ^ s[15];
    out[8] = s[8] ^ cv[0];
    out[9] = s[9] ^ cv[1];
    out[10] = s[10] ^ cv[2];
    out[11] = s[11] ^ cv[3];
    out[12] = s[12] ^ cv[4];
    out[13] = s[13] ^ cv[5];
    out[14] = s[14] ^ cv[6];
    out[15] = s[15] ^ cv[7];
}

// Single-chunk BLAKE3 of `len` (<=1024) bytes from private buffer `data`.
// Writes `out_len` (<=64) bytes little-endian into `out`.
static inline void blake3_hash_chunk(uchar const* data, uint len, uint out_len, uchar* out)
{
    uint cv[8] = { B3_IV0, B3_IV1, B3_IV2, B3_IV3, B3_IV4, B3_IV5, B3_IV6, B3_IV7 };

    uint nblocks = (len + 63u) / 64u;
    if (nblocks == 0u)
    {
        nblocks = 1u;
    }

    uint outwords[16];
    for (uint b = 0u; b < nblocks; ++b)
    {
        uint const base = b * 64u;
        uint const blen = (b == nblocks - 1u) ? (len - base) : 64u;

        uint m[16];
        for (uint w = 0u; w < 16u; ++w)
        {
            uint v = 0u;
            for (uint k = 0u; k < 4u; ++k)
            {
                uint const idx = base + w * 4u + k;
                uint const byte = (idx < len) ? (uint)data[idx] : 0u;
                v |= byte << (8u * k);
            }
            m[w] = v;
        }

        uint flags = 0u;
        if (b == 0u)
        {
            flags |= B3_FLAG_CHUNK_START;
        }
        if (b == nblocks - 1u)
        {
            flags |= B3_FLAG_CHUNK_END | B3_FLAG_ROOT;
        }

        uint out16[16];
        blake3_compress(cv, m, 0u, 0u, blen, flags, out16);

        if (b == nblocks - 1u)
        {
            for (uint i = 0u; i < 16u; ++i)
            {
                outwords[i] = out16[i];
            }
        }
        else
        {
            for (uint i = 0u; i < 8u; ++i)
            {
                cv[i] = out16[i];
            }
        }
    }

    for (uint i = 0u; i < out_len; ++i)
    {
        out[i] = (uchar)(outwords[i >> 2] >> (8u * (i & 3u)));
    }
}

#endif // LM_BLAKE3_CL
