///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
#include <algo/random_x/argon2d.hpp>
#include <algo/random_x/cuda/random_x.cuh>
#include <algo/random_x/superscalar.hpp>
#include <common/custom.hpp>
#include <common/error/cuda_error.hpp>


///////////////////////////////////////////////////////////////////////////////
// RandomX constants
///////////////////////////////////////////////////////////////////////////////
constexpr uint64_t RX_DATASET_ITEMS      { 34078720ull };
constexpr uint64_t RX_CACHE_ITEMS        { 4194304ull };
constexpr uint64_t RX_CACHE_BYTES        { RX_CACHE_ITEMS * 64ull };
constexpr uint32_t RX_SCRATCHPAD_L3      { 2097152u };
constexpr uint32_t RX_SCRATCHPAD_L2      { 262144u };
constexpr uint32_t RX_SCRATCHPAD_L1      { 16384u };
// Per-thread scratchpad stride includes 64-byte padding to avoid OOB reads.
// The reference allocates ScratchpadSize + 64 (vm_interpreted.cpp).
// With MASK_L3_8=0x1FFFF8, the last valid 8-byte-aligned address is 0x1FFFF8=2097144;
// the r[7] read then lands at 2097144+56=2097200, i.e. 48 bytes past 2097152.
constexpr uint32_t RX_SCRATCHPAD_STRIDE  { RX_SCRATCHPAD_L3 + 64u };
constexpr uint32_t RX_PROGRAM_COUNT      { 8u };
constexpr uint32_t RX_PROGRAM_ITERATIONS { 2048u };
constexpr uint32_t RX_PROGRAM_SIZE       { 256u };
constexpr uint32_t RX_JUMP_OFFSET        { 8u };
constexpr uint64_t RX_LCG_MUL           { 6364136223846793005ULL };
constexpr uint64_t RX_LCG_ADD           { 1442695040888963407ULL };

// Scratchpad address masks
constexpr uint32_t MASK_L3_8  { 0x1FFFF8u };
constexpr uint32_t MASK_L2_8  { 0x03FFF8u };
constexpr uint32_t MASK_L1_8  { 0x003FF8u };
constexpr uint32_t MASK_L3_64 { 0x1FFFC0u };

// Monero blob nonce offset
constexpr uint32_t RX_BLOB_NONCE_OFFSET { 39u };
constexpr uint32_t RX_BLOB_SIZE         { 77u };


///////////////////////////////////////////////////////////////////////////////
// GPU constants updated per-job
///////////////////////////////////////////////////////////////////////////////
__device__ __constant__ uint8_t  rx_blob[RX_BLOB_SIZE];
__device__ __constant__ uint32_t rx_target;
__device__ __constant__ uint64_t rx_start_nonce;


///////////////////////////////////////////////////////////////////////////////
// Blake2b
///////////////////////////////////////////////////////////////////////////////

__device__ __constant__
uint64_t RX_B2B_IV[8]
{
    0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
    0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
    0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};

__device__ __constant__
uint8_t RX_SIGMA[10][16]
{
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
};

__device__ __forceinline__
uint64_t rx_ror64(uint64_t const x, uint32_t const n)
{
    return (x >> n) | (x << (64u - n));
}

__device__ __forceinline__
void rx_b2b_G(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d,
              uint64_t const x, uint64_t const y)
{
    a = a + b + x;
    d = rx_ror64(d ^ a, 32u);
    c = c + d;
    b = rx_ror64(b ^ c, 24u);
    a = a + b + y;
    d = rx_ror64(d ^ a, 16u);
    c = c + d;
    b = rx_ror64(b ^ c, 63u);
}

__device__ __forceinline__
void rx_b2b_compress(uint64_t h[8], uint64_t const m[16],
                     uint64_t const t, uint32_t const last)
{
    uint64_t v[16];
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        v[i]     = h[i];
        v[i + 8] = RX_B2B_IV[i];
    }
    v[12] ^= t;
    if (0u != last) { v[14] ^= 0xFFFFFFFFFFFFFFFFULL; }
    for (uint32_t r{ 0u }; r < 12u; ++r)
    {
        uint8_t const* const s{ RX_SIGMA[r % 10u] };
        rx_b2b_G(v[0], v[4], v[8],  v[12], m[s[0]],  m[s[1]]);
        rx_b2b_G(v[1], v[5], v[9],  v[13], m[s[2]],  m[s[3]]);
        rx_b2b_G(v[2], v[6], v[10], v[14], m[s[4]],  m[s[5]]);
        rx_b2b_G(v[3], v[7], v[11], v[15], m[s[6]],  m[s[7]]);
        rx_b2b_G(v[0], v[5], v[10], v[15], m[s[8]],  m[s[9]]);
        rx_b2b_G(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        rx_b2b_G(v[2], v[7], v[8],  v[13], m[s[12]], m[s[13]]);
        rx_b2b_G(v[3], v[4], v[9],  v[14], m[s[14]], m[s[15]]);
    }
    for (uint32_t i{ 0u }; i < 8u; ++i) { h[i] ^= v[i] ^ v[i + 8u]; }
}

// Blake2b-512 of a blob (len <= 128 bytes, single block)
__device__
void rx_blake2b_512_blob(uint8_t const* const data, uint32_t const len, uint64_t h[8])
{
    for (uint32_t i{ 0u }; i < 8u; ++i) { h[i] = RX_B2B_IV[i]; }
    h[0] ^= 0x0000000001010040ULL;

    uint64_t m[16]{};
    for (uint32_t i{ 0u }; i < len; ++i)
    {
        m[i / 8u] |= (uint64_t)data[i] << ((i % 8u) * 8u);
    }

    rx_b2b_compress(h, m, (uint64_t)len, 1u);
}

// Blake2b-512 of a 64-bit nonce (used by cache build kernel)
__device__
void rx_blake2b_512_nonce(uint64_t const nonce, uint64_t h[8])
{
    for (uint32_t i{ 0u }; i < 8u; ++i) { h[i] = RX_B2B_IV[i]; }
    h[0] ^= 0x0000000001010040ULL;

    uint64_t m[16]{};
    m[0] = nonce;
    rx_b2b_compress(h, m, 8ull, 1u);
}

// Blake2b-256 of the 256-byte register file
__device__
void rx_blake2b_256_regfile(uint64_t const* const regfile, uint64_t out[4])
{
    uint64_t h[8];
    for (uint32_t i{ 0u }; i < 8u; ++i) { h[i] = RX_B2B_IV[i]; }
    h[0] ^= 0x0000000001010020ULL;

    uint64_t m[16];
    for (uint32_t i{ 0u }; i < 16u; ++i) { m[i] = regfile[i]; }
    rx_b2b_compress(h, m, 128ull, 0u);

    for (uint32_t i{ 0u }; i < 16u; ++i) { m[i] = regfile[16u + i]; }
    rx_b2b_compress(h, m, 256ull, 1u);

    out[0] = h[0]; out[1] = h[1]; out[2] = h[2]; out[3] = h[3];
}

// Blake2b-512 of 64 bytes (register file rehash between programs)
__device__
void rx_blake2b_512_buf64(uint64_t const in[8], uint64_t out[8])
{
    for (uint32_t i{ 0u }; i < 8u; ++i) { out[i] = RX_B2B_IV[i]; }
    out[0] ^= 0x0000000001010040ULL;

    uint64_t m[16]{};
    for (uint32_t i{ 0u }; i < 8u; ++i) { m[i] = in[i]; }
    rx_b2b_compress(out, m, 64ull, 1u);
}


// Blake2b-512 of 256 bytes — used to reseed AesGenerator4R between programs.
// Hashes the full register file (r[8] + f[4][2] + e[4][2] + a[4][2] = 32 uint64s = 256 bytes).
__device__
void rx_blake2b_512_regfile(uint64_t const in[32], uint64_t out[8])
{
    for (uint32_t i{ 0u }; i < 8u; ++i) { out[i] = RX_B2B_IV[i]; }
    out[0] ^= 0x0000000001010040ULL;

    uint64_t m[16];
    for (uint32_t i{ 0u }; i < 16u; ++i) { m[i] = in[i]; }
    rx_b2b_compress(out, m, 128ull, 0u);

    for (uint32_t i{ 0u }; i < 16u; ++i) { m[i] = in[16u + i]; }
    rx_b2b_compress(out, m, 256ull, 1u);
}


///////////////////////////////////////////////////////////////////////////////
// AES tables
///////////////////////////////////////////////////////////////////////////////

__device__ __constant__
uint8_t AES_SBOX[256]
{
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

__device__ __constant__
uint8_t AES_INVSBOX[256]
{
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};

__device__ __forceinline__
uint8_t aes_xtime(uint8_t const b)
{
    return static_cast<uint8_t>((b << 1u) ^ ((b >> 7u) ? 0x1bu : 0x00u));
}

__device__ __forceinline__
void aes_sub_bytes(uint8_t s[16])
{
    for (uint32_t i{ 0u }; i < 16u; ++i) { s[i] = AES_SBOX[s[i]]; }
}

__device__ __forceinline__
void aes_inv_sub_bytes(uint8_t s[16])
{
    for (uint32_t i{ 0u }; i < 16u; ++i) { s[i] = AES_INVSBOX[s[i]]; }
}

__device__ __forceinline__
void aes_shift_rows(uint8_t s[16])
{
    uint8_t t;
    t = s[1]; s[1] = s[5]; s[5] = s[9]; s[9] = s[13]; s[13] = t;
    t = s[2]; s[2] = s[10]; s[10] = t;
    t = s[6]; s[6] = s[14]; s[14] = t;
    t = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = s[3]; s[3] = t;
}

__device__ __forceinline__
void aes_inv_shift_rows(uint8_t s[16])
{
    uint8_t t;
    t = s[13]; s[13] = s[9]; s[9] = s[5]; s[5] = s[1]; s[1] = t;
    t = s[2]; s[2] = s[10]; s[10] = t;
    t = s[6]; s[6] = s[14]; s[14] = t;
    t = s[3]; s[3] = s[7]; s[7] = s[11]; s[11] = s[15]; s[15] = t;
}

__device__ __forceinline__
void aes_mix_column(uint8_t* const col)
{
    uint8_t const a0{ col[0] }, a1{ col[1] }, a2{ col[2] }, a3{ col[3] };
    col[0] = static_cast<uint8_t>(aes_xtime(a0) ^ aes_xtime(a1) ^ a1 ^ a2 ^ a3);
    col[1] = static_cast<uint8_t>(a0 ^ aes_xtime(a1) ^ aes_xtime(a2) ^ a2 ^ a3);
    col[2] = static_cast<uint8_t>(a0 ^ a1 ^ aes_xtime(a2) ^ aes_xtime(a3) ^ a3);
    col[3] = static_cast<uint8_t>(aes_xtime(a0) ^ a0 ^ a1 ^ a2 ^ aes_xtime(a3));
}

__device__ __forceinline__
uint8_t aes_gf_mul(uint8_t a, uint8_t b)
{
    uint8_t p{ 0u };
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        if (b & 1u) { p ^= a; }
        uint8_t const hi{ static_cast<uint8_t>(a & 0x80u) };
        a = static_cast<uint8_t>(a << 1u);
        if (hi) { a ^= 0x1bu; }
        b >>= 1u;
    }
    return p;
}

__device__ __forceinline__
void aes_inv_mix_column(uint8_t* const col)
{
    uint8_t const a0{ col[0] }, a1{ col[1] }, a2{ col[2] }, a3{ col[3] };
    col[0] = static_cast<uint8_t>(aes_gf_mul(0x0eu, a0) ^ aes_gf_mul(0x0bu, a1) ^ aes_gf_mul(0x0du, a2) ^ aes_gf_mul(0x09u, a3));
    col[1] = static_cast<uint8_t>(aes_gf_mul(0x09u, a0) ^ aes_gf_mul(0x0eu, a1) ^ aes_gf_mul(0x0bu, a2) ^ aes_gf_mul(0x0du, a3));
    col[2] = static_cast<uint8_t>(aes_gf_mul(0x0du, a0) ^ aes_gf_mul(0x09u, a1) ^ aes_gf_mul(0x0eu, a2) ^ aes_gf_mul(0x0bu, a3));
    col[3] = static_cast<uint8_t>(aes_gf_mul(0x0bu, a0) ^ aes_gf_mul(0x0du, a1) ^ aes_gf_mul(0x09u, a2) ^ aes_gf_mul(0x0eu, a3));
}

__device__ __forceinline__
void aes_mix_columns(uint8_t s[16])
{
    aes_mix_column(s + 0u);
    aes_mix_column(s + 4u);
    aes_mix_column(s + 8u);
    aes_mix_column(s + 12u);
}

__device__ __forceinline__
void aes_inv_mix_columns(uint8_t s[16])
{
    aes_inv_mix_column(s + 0u);
    aes_inv_mix_column(s + 4u);
    aes_inv_mix_column(s + 8u);
    aes_inv_mix_column(s + 12u);
}

__device__ __forceinline__
void aes_add_round_key(uint8_t s[16], uint8_t const key[16])
{
    for (uint32_t i{ 0u }; i < 16u; ++i) { s[i] ^= key[i]; }
}

__device__ __forceinline__
void aes_enc_round(uint8_t s[16], uint8_t const key[16])
{
    aes_sub_bytes(s);
    aes_shift_rows(s);
    aes_mix_columns(s);
    aes_add_round_key(s, key);
}

__device__ __forceinline__
void aes_dec_round(uint8_t s[16], uint8_t const key[16])
{
    aes_inv_sub_bytes(s);
    aes_inv_shift_rows(s);
    aes_inv_mix_columns(s);
    aes_add_round_key(s, key);
}


///////////////////////////////////////////////////////////////////////////////
// AES Generator keys (fixed RandomX spec constants)
///////////////////////////////////////////////////////////////////////////////

__device__ __constant__
uint8_t GEN1R_KEYS[4][16]
{
    { 0x53, 0xa5, 0xac, 0x6d, 0x09, 0x66, 0x71, 0x62, 0x2b, 0x55, 0xb5, 0xdb, 0x17, 0x49, 0xf4, 0xb4 },
    { 0x07, 0xaf, 0x7c, 0x6d, 0x0d, 0x71, 0x6a, 0x84, 0x78, 0xd3, 0x25, 0x17, 0x4e, 0xdc, 0xa1, 0x0d },
    { 0xf1, 0x62, 0x12, 0x3f, 0xc6, 0x7e, 0x94, 0x9f, 0x4f, 0x79, 0xc0, 0xf4, 0x45, 0xe3, 0x20, 0x3e },
    { 0x35, 0x81, 0xef, 0x6a, 0x7c, 0x31, 0xba, 0xb1, 0x88, 0x4c, 0x31, 0x16, 0x54, 0x91, 0x16, 0x49 }
};

__device__ __constant__
uint8_t GEN4R_KEYS_A[4][16]
{
    { 0xdd, 0xaa, 0x21, 0x64, 0xdb, 0x3d, 0x83, 0xd1, 0x2b, 0x6d, 0x54, 0x2f, 0x3f, 0xd2, 0xe5, 0x99 },
    { 0x50, 0x34, 0x0e, 0xb2, 0x55, 0x3f, 0x91, 0xb6, 0x53, 0x9d, 0xf7, 0x06, 0xe5, 0xcd, 0xdf, 0xa5 },
    { 0x04, 0xd9, 0x3e, 0x5c, 0xaf, 0x7b, 0x5e, 0x51, 0x9f, 0x67, 0xa4, 0x0a, 0xbf, 0x02, 0x1c, 0x17 },
    { 0x63, 0x37, 0x62, 0x85, 0x08, 0x5d, 0x8f, 0xe7, 0x85, 0x37, 0x67, 0xcd, 0x91, 0xd2, 0xde, 0xd8 }
};

__device__ __constant__
uint8_t GEN4R_KEYS_B[4][16]
{
    { 0x73, 0x6f, 0x82, 0xb5, 0xa6, 0xa7, 0xd6, 0xe3, 0x6d, 0x8b, 0x51, 0x3d, 0xb4, 0xff, 0x9e, 0x22 },
    { 0xf3, 0x6b, 0x56, 0xc7, 0xd9, 0xb3, 0x10, 0x9c, 0x4e, 0x4d, 0x02, 0xe9, 0xd2, 0xb7, 0x72, 0xb2 },
    { 0xe7, 0xc9, 0x73, 0xf2, 0x8b, 0xa3, 0x65, 0xf7, 0x0a, 0x66, 0xa9, 0x2b, 0xa7, 0xef, 0x3b, 0xf6 },
    { 0x09, 0xd6, 0x7c, 0x7a, 0xde, 0x39, 0x58, 0x91, 0xfd, 0xd1, 0x06, 0x0c, 0x2d, 0x76, 0xb0, 0xc0 }
};

__device__ __constant__
uint8_t AESHASH1R_STATE[4][16]
{
    { 0x0d, 0x2c, 0xb5, 0x92, 0xde, 0x56, 0xa8, 0x9f, 0x47, 0xdb, 0x82, 0xcc, 0xad, 0x3a, 0x98, 0xd7 },
    { 0x6e, 0x99, 0x8d, 0x33, 0x98, 0xb7, 0xc7, 0x15, 0x5a, 0x12, 0x9e, 0xf5, 0x57, 0x80, 0xe7, 0xac },
    { 0x17, 0x00, 0x77, 0x6a, 0xd0, 0xc7, 0x62, 0xae, 0x6b, 0x50, 0x79, 0x50, 0xe4, 0x7c, 0xa0, 0xe8 },
    { 0x0c, 0x24, 0x0a, 0x63, 0x8d, 0x82, 0xad, 0x07, 0x05, 0x00, 0xa1, 0x79, 0x48, 0x49, 0x99, 0x7e }
};

__device__ __constant__
uint8_t AESHASH1R_XKEYS[2][16]
{
    { 0x89, 0x83, 0xfa, 0xf6, 0x9f, 0x94, 0x24, 0x8b, 0xbf, 0x56, 0xdc, 0x90, 0x01, 0x02, 0x89, 0x06 },
    { 0xd1, 0x63, 0xb2, 0x61, 0x3c, 0xe0, 0xf4, 0x51, 0xc6, 0x43, 0x10, 0xee, 0x9b, 0xf9, 0x18, 0xed }
};


///////////////////////////////////////////////////////////////////////////////
// AES Generators
///////////////////////////////////////////////////////////////////////////////

__device__
void aes_gen1r_fill(uint8_t state[64], uint8_t* const dest, uint32_t const count)
{
    uint8_t* col0{ state + 0u };
    uint8_t* col1{ state + 16u };
    uint8_t* col2{ state + 32u };
    uint8_t* col3{ state + 48u };

    for (uint32_t i{ 0u }; i < count; ++i)
    {
        aes_dec_round(col0, GEN1R_KEYS[0]);
        aes_enc_round(col1, GEN1R_KEYS[1]);
        aes_dec_round(col2, GEN1R_KEYS[2]);
        aes_enc_round(col3, GEN1R_KEYS[3]);

        uint8_t* const out{ dest + (uint64_t)i * 64u };
        for (uint32_t b{ 0u }; b < 16u; ++b) { out[b]      = col0[b]; }
        for (uint32_t b{ 0u }; b < 16u; ++b) { out[16u + b] = col1[b]; }
        for (uint32_t b{ 0u }; b < 16u; ++b) { out[32u + b] = col2[b]; }
        for (uint32_t b{ 0u }; b < 16u; ++b) { out[48u + b] = col3[b]; }
    }
}

__device__
void aes_gen4r_fill(uint8_t state[64], uint8_t* const dest, uint32_t const count)
{
    uint8_t* col0{ state + 0u };
    uint8_t* col1{ state + 16u };
    uint8_t* col2{ state + 32u };
    uint8_t* col3{ state + 48u };

    for (uint32_t i{ 0u }; i < count; ++i)
    {
        for (uint32_t r{ 0u }; r < 4u; ++r)
        {
            aes_dec_round(col0, GEN4R_KEYS_A[r]);
            aes_enc_round(col1, GEN4R_KEYS_A[r]);
            aes_dec_round(col2, GEN4R_KEYS_B[r]);
            aes_enc_round(col3, GEN4R_KEYS_B[r]);
        }
        uint8_t* const out{ dest + (uint64_t)i * 64u };
        for (uint32_t b{ 0u }; b < 16u; ++b) { out[b]      = col0[b]; }
        for (uint32_t b{ 0u }; b < 16u; ++b) { out[16u + b] = col1[b]; }
        for (uint32_t b{ 0u }; b < 16u; ++b) { out[32u + b] = col2[b]; }
        for (uint32_t b{ 0u }; b < 16u; ++b) { out[48u + b] = col3[b]; }
    }
}

__device__
void aes_hash1r(uint8_t const* const scratchpad, uint64_t out[8])
{
    uint8_t s0[16], s1[16], s2[16], s3[16];
    for (uint32_t i{ 0u }; i < 16u; ++i) { s0[i] = AESHASH1R_STATE[0][i]; }
    for (uint32_t i{ 0u }; i < 16u; ++i) { s1[i] = AESHASH1R_STATE[1][i]; }
    for (uint32_t i{ 0u }; i < 16u; ++i) { s2[i] = AESHASH1R_STATE[2][i]; }
    for (uint32_t i{ 0u }; i < 16u; ++i) { s3[i] = AESHASH1R_STATE[3][i]; }

    uint32_t const num_blocks{ RX_SCRATCHPAD_L3 / 64u };
    for (uint32_t blk{ 0u }; blk < num_blocks; ++blk)
    {
        uint8_t const* const k{ scratchpad + (uint64_t)blk * 64u };
        aes_enc_round(s0, k + 0u);
        aes_dec_round(s1, k + 16u);
        aes_enc_round(s2, k + 32u);
        aes_dec_round(s3, k + 48u);
    }

    aes_enc_round(s0, AESHASH1R_XKEYS[0]);
    aes_dec_round(s1, AESHASH1R_XKEYS[0]);
    aes_enc_round(s2, AESHASH1R_XKEYS[0]);
    aes_dec_round(s3, AESHASH1R_XKEYS[0]);

    aes_enc_round(s0, AESHASH1R_XKEYS[1]);
    aes_dec_round(s1, AESHASH1R_XKEYS[1]);
    aes_enc_round(s2, AESHASH1R_XKEYS[1]);
    aes_dec_round(s3, AESHASH1R_XKEYS[1]);

    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        uint8_t const* const src{ (i < 2u) ? s0 : (i < 4u) ? s1 : (i < 6u) ? s2 : s3 };
        uint32_t const off{ (i % 2u) * 8u };
        out[i] = (uint64_t)src[off + 0u]          | ((uint64_t)src[off + 1u] << 8u)  |
                 ((uint64_t)src[off + 2u] << 16u)  | ((uint64_t)src[off + 3u] << 24u) |
                 ((uint64_t)src[off + 4u] << 32u)  | ((uint64_t)src[off + 5u] << 40u) |
                 ((uint64_t)src[off + 6u] << 48u)  | ((uint64_t)src[off + 7u] << 56u);
    }
}


///////////////////////////////////////////////////////////////////////////////
// VM helpers
///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__
uint64_t sp_read_u64(uint8_t const* const sp, uint32_t const addr)
{
    uint8_t const* const p{ sp + addr };
    return (uint64_t)p[0]          | ((uint64_t)p[1] << 8u)  |
           ((uint64_t)p[2] << 16u) | ((uint64_t)p[3] << 24u) |
           ((uint64_t)p[4] << 32u) | ((uint64_t)p[5] << 40u) |
           ((uint64_t)p[6] << 48u) | ((uint64_t)p[7] << 56u);
}

__device__ __forceinline__
void sp_write_u64(uint8_t* const sp, uint32_t const addr, uint64_t const val)
{
    uint8_t* const p{ sp + addr };
    p[0] = (uint8_t)val;           p[1] = (uint8_t)(val >> 8u);
    p[2] = (uint8_t)(val >> 16u);  p[3] = (uint8_t)(val >> 24u);
    p[4] = (uint8_t)(val >> 32u);  p[5] = (uint8_t)(val >> 40u);
    p[6] = (uint8_t)(val >> 48u);  p[7] = (uint8_t)(val >> 56u);
}

__device__ __forceinline__
uint32_t sp_mem_read_mask(uint8_t const src, uint8_t const dst, uint8_t const mod)
{
    if (src == dst) { return MASK_L3_8; }
    return (mod & 3u) ? MASK_L1_8 : MASK_L2_8;
}

__device__ __forceinline__
uint32_t sp_store_mask(uint8_t const mod)
{
    if ((mod >> 4u) >= 14u) { return MASK_L3_8; }
    return (mod & 3u) ? MASK_L1_8 : MASK_L2_8;
}

__device__ __forceinline__
double bytes_to_f_double(uint8_t const* const b)
{
    int32_t const i{ (int32_t)((uint32_t)b[0] | ((uint32_t)b[1] << 8u) |
                               ((uint32_t)b[2] << 16u) | ((uint32_t)b[3] << 24u)) };
    return (double)i;
}

__device__ __forceinline__
double bytes_to_e_double(uint8_t const* const b, uint64_t const eMask)
{
    int32_t const i{ (int32_t)((uint32_t)b[0] | ((uint32_t)b[1] << 8u) |
                               ((uint32_t)b[2] << 16u) | ((uint32_t)b[3] << 24u)) };
    double d{ (double)i };
    uint64_t bits;
    __builtin_memcpy(&bits, &d, sizeof(bits));
    bits &= 0x00FFFFFFFFFFFFFFull;  // dynamicMantissaMask: keep bits 0-55, zero sign+top exponent
    bits |= eMask;
    __builtin_memcpy(&d, &bits, sizeof(d));
    return d;
}

__device__ __forceinline__
uint64_t rx_reciprocal(uint32_t const divisor)
{
    // Compute floor(2^x / divisor) for the highest x such that the result < 2^64.
    // Matches randomx_reciprocal() from the reference: 2^(63 + shift) / divisor
    // where shift = 64 - clz64(divisor).
    uint64_t const d{ (uint64_t)divisor };
    uint64_t const p{ 1ULL << 63 };
    uint64_t const q{ p / d };
    uint64_t const r{ p % d };
    uint32_t const shift{ 64u - (uint32_t)__clzll(d) };
    return (q << shift) + ((r << shift) / d);
}


///////////////////////////////////////////////////////////////////////////////
// VM program execution
///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__
double rx_fp_add(double a, double b, uint32_t fprc)
{
    if (fprc == 1u) { return __dadd_rd(a, b); }
    if (fprc == 2u) { return __dadd_ru(a, b); }
    if (fprc == 3u) { return __dadd_rz(a, b); }
    return __dadd_rn(a, b);
}


__device__ __forceinline__
double rx_fp_sub(double a, double b, uint32_t fprc)
{
    if (fprc == 1u) { return __dsub_rd(a, b); }
    if (fprc == 2u) { return __dsub_ru(a, b); }
    if (fprc == 3u) { return __dsub_rz(a, b); }
    return __dsub_rn(a, b);
}


__device__ __forceinline__
double rx_fp_mul(double a, double b, uint32_t fprc)
{
    if (fprc == 1u) { return __dmul_rd(a, b); }
    if (fprc == 2u) { return __dmul_ru(a, b); }
    if (fprc == 3u) { return __dmul_rz(a, b); }
    return __dmul_rn(a, b);
}


__device__ __forceinline__
double rx_fp_div(double a, double b, uint32_t fprc)
{
    if (fprc == 1u) { return __ddiv_rd(a, b); }
    if (fprc == 2u) { return __ddiv_ru(a, b); }
    if (fprc == 3u) { return __ddiv_rz(a, b); }
    return __ddiv_rn(a, b);
}


__device__ __forceinline__
double rx_fp_sqrt(double a, uint32_t fprc)
{
    if (fprc == 1u) { return __dsqrt_rd(a); }
    if (fprc == 2u) { return __dsqrt_ru(a); }
    if (fprc == 3u) { return __dsqrt_rz(a); }
    return __dsqrt_rn(a);
}


struct RxProgram
{
    uint8_t  opcode[RX_PROGRAM_SIZE];
    uint8_t  dst[RX_PROGRAM_SIZE];
    uint8_t  src[RX_PROGRAM_SIZE];
    uint8_t  mod[RX_PROGRAM_SIZE];
    uint32_t imm32[RX_PROGRAM_SIZE];
    uint32_t target[RX_PROGRAM_SIZE];
};

__device__
void rx_parse_program(uint8_t const* const prog_bytes, RxProgram& prog)
{
    for (uint32_t i{ 0u }; i < RX_PROGRAM_SIZE; ++i)
    {
        uint8_t const* const p{ prog_bytes + (uint64_t)i * 8u };
        prog.opcode[i] = p[0];
        prog.dst[i]    = p[1];
        prog.src[i]    = p[2];
        prog.mod[i]    = p[3];
        prog.imm32[i]  = (uint32_t)p[4] | ((uint32_t)p[5] << 8u) |
                         ((uint32_t)p[6] << 16u) | ((uint32_t)p[7] << 24u);
    }

    // Compile pass: compute CBRANCH targets using registerUsage (mirrors reference JIT compile).
    // registerUsage[r] = last instruction that modified r, or -1 if never modified.
    // CBRANCH jumps to registerUsage[d]+1 (reference: pc=target, then ++pc in for-loop).
    int32_t regUsage[8];
    for (uint32_t j{ 0u }; j < 8u; ++j) { regUsage[j] = -1; }

    for (uint32_t i{ 0u }; i < RX_PROGRAM_SIZE; ++i)
    {
        uint8_t  const opc{ prog.opcode[i] };
        uint8_t  const di { prog.dst[i] };
        uint8_t  const si { prog.src[i] };
        uint32_t const imm{ prog.imm32[i] };

        if (opc < 76u)
        {
            regUsage[di & 7u] = (int32_t)i;
        }
        else if (opc < 84u)
        {
            // IMUL_RCP: NOP when divisor is 0 or a power of two
            uint32_t const u{ imm };
            if (u != 0u && (u & (u - 1u)) != 0u) { regUsage[di & 7u] = (int32_t)i; }
        }
        else if (opc < 116u)
        {
            regUsage[di & 7u] = (int32_t)i;
        }
        else if (opc < 120u)
        {
            // ISWAP_R: updates both registers when dst != src
            uint8_t const d{ di & 7u }, s{ si & 7u };
            if (d != s) { regUsage[d] = (int32_t)i; regUsage[s] = (int32_t)i; }
        }
        else if (opc >= 214u && opc < 239u)
        {
            // CBRANCH: target = registerUsage[d]+1; then marks all registers modified
            uint8_t const d{ di & 7u };
            prog.target[i] = (uint32_t)(regUsage[d] + 1);
            for (uint32_t j{ 0u }; j < 8u; ++j) { regUsage[j] = (int32_t)i; }
        }
    }
}

__device__
void rx_execute_program(
    uint64_t        r[8],
    double          f[4][2],
    double          e[4][2],
    double const    a[4][2],
    uint8_t* const  sp,
    RxProgram const& prog,
    uint64_t const  eMask_lo,
    uint64_t const  eMask_hi,
    uint32_t&       fprc)
{
    uint32_t ip{ 0u };
    while (ip < RX_PROGRAM_SIZE)
    {
        uint8_t const  opc { prog.opcode[ip] };
        uint8_t const  di  { prog.dst[ip] };
        uint8_t const  si  { prog.src[ip] };
        uint8_t const  mod { prog.mod[ip] };
        uint32_t const imm { prog.imm32[ip] };
        int64_t  const imm_s{ (int64_t)(int32_t)imm };

        if (opc < 16u)
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint8_t const shift{ (mod >> 2u) & 3u };
            r[d] += r[s] << shift;
            if (d == 5u) { r[d] += (uint64_t)(int64_t)(int32_t)imm; }
        }
        else if (opc < 23u)
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const mask{ sp_mem_read_mask(s, d, mod) };
            uint64_t const base{ (s == d) ? 0ull : r[s] };
            uint32_t const addr{ static_cast<uint32_t>(base + (uint64_t)imm_s) & mask };
            r[d] += sp_read_u64(sp, addr);
        }
        else if (opc < 39u)
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            r[d] -= (d == s) ? (uint64_t)imm_s : r[s];
        }
        else if (opc < 46u)
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const mask{ sp_mem_read_mask(s, d, mod) };
            uint64_t const base{ (s == d) ? 0ull : r[s] };
            uint32_t const addr{ static_cast<uint32_t>(base + (uint64_t)imm_s) & mask };
            r[d] -= sp_read_u64(sp, addr);
        }
        else if (opc < 62u)
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            r[d] *= (d == s) ? (uint64_t)imm_s : r[s];
        }
        else if (opc < 66u)
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const mask{ sp_mem_read_mask(s, d, mod) };
            uint64_t const base{ (s == d) ? 0ull : r[s] };
            uint32_t const addr{ static_cast<uint32_t>(base + (uint64_t)imm_s) & mask };
            r[d] *= sp_read_u64(sp, addr);
        }
        else if (opc < 70u)
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            unsigned __int128 const prod{ (unsigned __int128)r[d] * (unsigned __int128)((d == s) ? r[d] : r[s]) };
            r[d] = (uint64_t)(prod >> 64u);
        }
        else if (opc < 71u)
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const mask{ sp_mem_read_mask(s, d, mod) };
            uint64_t const base{ (s == d) ? 0ull : r[s] };
            uint32_t const addr{ static_cast<uint32_t>(base + (uint64_t)imm_s) & mask };
            unsigned __int128 const prod{ (unsigned __int128)r[d] * (unsigned __int128)sp_read_u64(sp, addr) };
            r[d] = (uint64_t)(prod >> 64u);
        }
        else if (opc < 75u)
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            __int128 const prod{ (__int128)(int64_t)r[d] * (__int128)(int64_t)((d == s) ? r[d] : r[s]) };
            r[d] = (uint64_t)((unsigned __int128)prod >> 64u);
        }
        else if (opc < 76u)
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const mask{ sp_mem_read_mask(s, d, mod) };
            uint64_t const base{ (s == d) ? 0ull : r[s] };
            uint32_t const addr{ static_cast<uint32_t>(base + (uint64_t)imm_s) & mask };
            __int128 const prod{ (__int128)(int64_t)r[d] * (__int128)(int64_t)sp_read_u64(sp, addr) };
            r[d] = (uint64_t)((unsigned __int128)prod >> 64u);
        }
        else if (opc < 84u)
        {
            uint8_t const d{ di & 7u };
            uint64_t const u{ (uint64_t)imm };
            bool const is_pow2{ (u != 0u) && ((u & (u - 1u)) == 0u) };
            if (u != 0u && !is_pow2)
            {
                r[d] *= rx_reciprocal(imm);
            }
        }
        else if (opc < 86u)
        {
            uint8_t const d{ di & 7u };
            r[d] = ~r[d] + 1u;
        }
        else if (opc < 101u)
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            r[d] ^= (d == s) ? (uint64_t)imm_s : r[s];
        }
        else if (opc < 106u)
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const mask{ sp_mem_read_mask(s, d, mod) };
            uint64_t const base{ (s == d) ? 0ull : r[s] };
            uint32_t const addr{ static_cast<uint32_t>(base + (uint64_t)imm_s) & mask };
            r[d] ^= sp_read_u64(sp, addr);
        }
        else if (opc < 114u)
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const n{ static_cast<uint32_t>((d == s) ? imm : r[s]) & 63u };
            r[d] = rx_ror64(r[d], n);
        }
        else if (opc < 116u)
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const n{ static_cast<uint32_t>((d == s) ? imm : r[s]) & 63u };
            r[d] = (r[d] << n) | (r[d] >> (64u - n));
        }
        else if (opc < 120u)
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            if (d != s)
            {
                uint64_t const tmp{ r[d] };
                r[d] = r[s]; r[s] = tmp;
            }
        }
        else if (opc < 124u)
        {
            uint8_t const idx{ di & 3u };
            if ((di & 4u) == 0u)
            {
                double const tmp{ f[idx][0] }; f[idx][0] = f[idx][1]; f[idx][1] = tmp;
            }
            else
            {
                double const tmp{ e[idx][0] }; e[idx][0] = e[idx][1]; e[idx][1] = tmp;
            }
        }
        else if (opc < 140u)
        {
            uint8_t const d{ di & 3u }, s{ si & 3u };
            f[d][0] = rx_fp_add(f[d][0], a[s][0], fprc);
            f[d][1] = rx_fp_add(f[d][1], a[s][1], fprc);
        }
        else if (opc < 145u)
        {
            uint8_t const d{ di & 3u }, s{ si & 7u };
            uint32_t const fmask{ (mod & 3u) ? MASK_L1_8 : MASK_L2_8 };
            uint32_t const addr{ static_cast<uint32_t>(r[s] + (uint64_t)imm_s) & fmask };
            uint8_t const* const mp{ sp + addr };
            f[d][0] = rx_fp_add(f[d][0], bytes_to_f_double(mp),      fprc);
            f[d][1] = rx_fp_add(f[d][1], bytes_to_f_double(mp + 4u), fprc);
        }
        else if (opc < 161u)
        {
            uint8_t const d{ di & 3u }, s{ si & 3u };
            f[d][0] = rx_fp_sub(f[d][0], a[s][0], fprc);
            f[d][1] = rx_fp_sub(f[d][1], a[s][1], fprc);
        }
        else if (opc < 166u)
        {
            uint8_t const d{ di & 3u }, s{ si & 7u };
            uint32_t const fmask{ (mod & 3u) ? MASK_L1_8 : MASK_L2_8 };
            uint32_t const addr{ static_cast<uint32_t>(r[s] + (uint64_t)imm_s) & fmask };
            uint8_t const* const mp{ sp + addr };
            f[d][0] = rx_fp_sub(f[d][0], bytes_to_f_double(mp),      fprc);
            f[d][1] = rx_fp_sub(f[d][1], bytes_to_f_double(mp + 4u), fprc);
        }
        else if (opc < 172u)
        {
            uint8_t const d{ di & 3u };
            constexpr uint64_t FSCAL_MASK{ 0x80F0000000000000ULL };
            uint64_t bits0, bits1;
            __builtin_memcpy(&bits0, &f[d][0], 8u);
            __builtin_memcpy(&bits1, &f[d][1], 8u);
            bits0 ^= FSCAL_MASK;
            bits1 ^= FSCAL_MASK;
            __builtin_memcpy(&f[d][0], &bits0, 8u);
            __builtin_memcpy(&f[d][1], &bits1, 8u);
        }
        else if (opc < 204u)
        {
            uint8_t const d{ di & 3u }, s{ si & 3u };
            e[d][0] = rx_fp_mul(e[d][0], a[s][0], fprc);
            e[d][1] = rx_fp_mul(e[d][1], a[s][1], fprc);
        }
        else if (opc < 208u)
        {
            uint8_t const d{ di & 3u }, s{ si & 7u };
            uint32_t const fmask{ (mod & 3u) ? MASK_L1_8 : MASK_L2_8 };
            uint32_t const addr{ static_cast<uint32_t>(r[s] + (uint64_t)imm_s) & fmask };
            uint8_t const* const mp{ sp + addr };
            double const dlo{ bytes_to_e_double(mp,      eMask_lo) };
            double const dhi{ bytes_to_e_double(mp + 4u, eMask_hi) };
            e[d][0] = rx_fp_div(e[d][0], dlo, fprc);
            e[d][1] = rx_fp_div(e[d][1], dhi, fprc);
        }
        else if (opc < 214u)
        {
            uint8_t const d{ di & 3u };
            e[d][0] = rx_fp_sqrt(e[d][0], fprc);
            e[d][1] = rx_fp_sqrt(e[d][1], fprc);
        }
        else if (opc < 239u)
        {
            uint8_t const d{ di & 7u };
            uint32_t const b{ (mod >> 4u) + RX_JUMP_OFFSET };
            int64_t cimm{ imm_s };
            cimm |= (int64_t)(1ull << b);
            if (b > 0u) { cimm &= ~(int64_t)(1ull << (b - 1u)); }

            uint32_t const cbTarget{ prog.target[ip] };
            r[d] = (uint64_t)((int64_t)r[d] + cimm);

            if (((r[d] >> b) & 0xFFu) == 0u)
            {
                ip = cbTarget;
                continue;
            }
        }
        else if (opc < 240u)
        {
            uint8_t const s{ si & 7u };
            fprc = static_cast<uint32_t>(rx_ror64(r[s], imm & 63u)) & 3u;
        }
        else
        {
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const mask{ sp_store_mask(mod) };
            uint32_t const addr{ static_cast<uint32_t>(r[d] + (uint64_t)imm_s) & mask };
            sp_write_u64(sp, addr, r[s]);
        }

        ip++;
    }
}


///////////////////////////////////////////////////////////////////////////////
// Shared VM execution body (used by both cache-build seed and miner)
///////////////////////////////////////////////////////////////////////////////

__device__
void rx_run_vm(
    uint64_t       seed[8],
    uint64_t const* const dataset,
    uint8_t* const  sp)
{
    // ── Step 1: init AES state from seed ─────────────────────────────────
    uint8_t aes_state[64];
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        uint64_t const w{ seed[i] };
        for (uint32_t b{ 0u }; b < 8u; ++b)
        {
            aes_state[i * 8u + b] = (uint8_t)(w >> (b * 8u));
        }
    }

    // ── Step 2: copy AES state for AesGenerator4R ────────────────────────
    // NOTE: AES-4R (program entropy) runs FIRST, then AES-1R (scratchpad fill)
    // runs SECOND from the state already modified by AES-4R.  This matches
    // VmBase::run() in the reference: generateProgram() calls fillAes4Rx4,
    // then fillAes1Rx4 is called on the same (now-modified) tempHash.
    uint8_t gen4r_state[64];
    for (uint32_t i{ 0u }; i < 64u; ++i) { gen4r_state[i] = aes_state[i]; }

    // ── Step 3: VM registers ──────────────────────────────────────────────
    uint64_t r[8]{};
    double   f[4][2]{};
    double   e[4][2]{};
    double   a[4][2]{};
    uint32_t ma{ 0u };
    uint32_t mx{ 0u };
    uint8_t  readReg[4]{ 0u, 2u, 4u, 6u };
    uint64_t eMask_lo{ 0x3C00000000000000ULL };
    uint64_t eMask_hi{ 0x3C00000000000000ULL };
    uint32_t fprc{ 0u };
    uint8_t  prog_buf[2176];
    RxProgram prog{};

    for (uint32_t prog_idx{ 0u }; prog_idx < RX_PROGRAM_COUNT; ++prog_idx)
    {
        // AES-4R: generate program entropy (modifies gen4r_state in place)
        aes_gen4r_fill(gen4r_state, prog_buf, 34u);

        // AES-1R: fill scratchpad from the NOW-modified state (first program only)
        if (prog_idx == 0u)
        {
            aes_gen1r_fill(gen4r_state, sp, RX_SCRATCHPAD_L3 / 64u);
        }

        uint64_t hdr[16];
        for (uint32_t i{ 0u }; i < 16u; ++i)
        {
            uint8_t const* const p{ prog_buf + (uint64_t)i * 8u };
            hdr[i] = (uint64_t)p[0]          | ((uint64_t)p[1] << 8u)  |
                     ((uint64_t)p[2] << 16u)  | ((uint64_t)p[3] << 24u) |
                     ((uint64_t)p[4] << 32u)  | ((uint64_t)p[5] << 40u) |
                     ((uint64_t)p[6] << 48u)  | ((uint64_t)p[7] << 56u);
        }

        for (uint32_t i{ 0u }; i < 4u; ++i)
        {
            uint64_t const qlo{ hdr[i * 2u] };
            uint64_t const qhi{ hdr[i * 2u + 1u] };
            uint64_t const frac_lo{ qlo & 0x000FFFFFFFFFFFFFULL };
            uint32_t const exp_lo { static_cast<uint32_t>((qlo >> 59u) & 0x1Fu) };
            uint64_t const bits_lo{ ((uint64_t)(exp_lo + 1023u) << 52u) | frac_lo };
            uint64_t const frac_hi{ qhi & 0x000FFFFFFFFFFFFFULL };
            uint32_t const exp_hi { static_cast<uint32_t>((qhi >> 59u) & 0x1Fu) };
            uint64_t const bits_hi{ ((uint64_t)(exp_hi + 1023u) << 52u) | frac_hi };
            __builtin_memcpy(&a[i][0], &bits_lo, 8u);
            __builtin_memcpy(&a[i][1], &bits_hi, 8u);
        }

        ma = static_cast<uint32_t>(hdr[8]) & 0xFFFFFFC0u;
        mx = static_cast<uint32_t>(hdr[10]);

        uint64_t const q12{ hdr[12] };
        readReg[0] = ((q12 >> 0u) & 1u) ? 1u : 0u;
        readReg[1] = ((q12 >> 1u) & 1u) ? 3u : 2u;
        readReg[2] = ((q12 >> 2u) & 1u) ? 5u : 4u;
        readReg[3] = ((q12 >> 3u) & 1u) ? 7u : 6u;

        // datasetOffset: per-program byte offset into dataset (spec section 7.5)
        // DatasetExtraItems = 33554368 / 64 = 524287, so modulus = 524288
        uint64_t const datasetOffset{ (hdr[13] % 524288ULL) * 64ULL };

        uint64_t const q14{ hdr[14] };
        uint64_t const q15{ hdr[15] };
        // getFloatMask: (entropy & 0x3FFFFF) | 0x3000000000000000 | ((entropy>>60)<<56)
        // constExponentBits=0x300 at bit 52 → 0x3000000000000000; dynamic 4 bits at 56-59
        eMask_lo = (q14 & 0x3FFFFFull) | 0x3000000000000000ULL | ((q14 >> 60u) << 56u);
        eMask_hi = (q15 & 0x3FFFFFull) | 0x3000000000000000ULL | ((q15 >> 60u) << 56u);

        rx_parse_program(prog_buf + 128u, prog);

        // Integer registers r[] carry over from previous program (or 0 for first).
        // Only the per-iteration scratchpad-address accumulators reset per program.
        uint32_t spAddr0{ mx };
        uint32_t spAddr1{ ma };

        for (uint32_t iter{ 0u }; iter < RX_PROGRAM_ITERATIONS; ++iter)
        {
            // reference: spMix = r[readReg0]^r[readReg1]; spAddr0 ^= low32(spMix); spAddr1 ^= high32(spMix)
            uint64_t const spMix{ r[readReg[0]] ^ r[readReg[1]] };
            spAddr0 ^= static_cast<uint32_t>(spMix);
            spAddr1 ^= static_cast<uint32_t>(spMix >> 32u);

            // reference ScratchpadL3Mask64 = (L3/8 - 1)*8 = 0x1FFFF8 = MASK_L3_8 (8-byte aligned)
            uint32_t const addr2{ spAddr0 & MASK_L3_8 };
            for (uint32_t i{ 0u }; i < 8u; ++i)
            {
                r[i] ^= sp_read_u64(sp, addr2 + i * 8u);
            }

            uint32_t const addr3{ spAddr1 & MASK_L3_8 };
            uint8_t const* const blk3{ sp + addr3 };
            for (uint32_t i{ 0u }; i < 4u; ++i)
            {
                uint8_t const* const lo{ blk3 + i * 8u };
                uint8_t const* const hi{ blk3 + 32u + i * 8u };
                f[i][0] = bytes_to_f_double(lo);
                f[i][1] = bytes_to_f_double(lo + 4u);
                e[i][0] = bytes_to_e_double(hi,      eMask_lo);
                e[i][1] = bytes_to_e_double(hi + 4u, eMask_hi);
            }

            rx_execute_program(r, f, e, a, sp, prog, eMask_lo, eMask_hi, fprc);

            uint32_t const rr23{ static_cast<uint32_t>(r[readReg[2]]) ^
                                  static_cast<uint32_t>(r[readReg[3]]) };
            mx ^= rr23;
            mx &= 0xFFFFFFC0u;

            uint64_t const readPtr{ datasetOffset + (static_cast<uint64_t>(ma) & 0x7FFFFFC0ULL) };
            uint64_t const* const dblk{ dataset + readPtr / 8ULL };
            for (uint32_t i{ 0u }; i < 8u; ++i)
            {
                r[i] ^= dblk[i];
            }

            { uint32_t const tmp{ mx }; mx = ma; ma = tmp; }

            uint32_t const addr9{ spAddr1 & MASK_L3_8 };
            for (uint32_t i{ 0u }; i < 8u; ++i) { sp_write_u64(sp, addr9 + i * 8u, r[i]); }

            for (uint32_t i{ 0u }; i < 4u; ++i)
            {
                uint64_t fb0, fb1, eb0, eb1;
                __builtin_memcpy(&fb0, &f[i][0], 8u); __builtin_memcpy(&eb0, &e[i][0], 8u);
                __builtin_memcpy(&fb1, &f[i][1], 8u); __builtin_memcpy(&eb1, &e[i][1], 8u);
                fb0 ^= eb0; fb1 ^= eb1;
                __builtin_memcpy(&f[i][0], &fb0, 8u); __builtin_memcpy(&f[i][1], &fb1, 8u);
            }

            uint32_t const addr11{ spAddr0 & MASK_L3_8 };
            for (uint32_t i{ 0u }; i < 4u; ++i)
            {
                uint64_t fb0, fb1;
                __builtin_memcpy(&fb0, &f[i][0], 8u);
                __builtin_memcpy(&fb1, &f[i][1], 8u);
                sp_write_u64(sp, addr11 + i * 16u,      fb0);
                sp_write_u64(sp, addr11 + i * 16u + 8u, fb1);
            }

            spAddr0 = 0u;
            spAddr1 = 0u;
        }

        if (prog_idx < RX_PROGRAM_COUNT - 1u)
        {
            uint64_t regfile[32]{};
            for (uint32_t i{ 0u }; i < 8u; ++i) { regfile[i] = r[i]; }
            for (uint32_t i{ 0u }; i < 4u; ++i)
            {
                __builtin_memcpy(&regfile[8u  + i * 2u],      &f[i][0], 8u);
                __builtin_memcpy(&regfile[8u  + i * 2u + 1u], &f[i][1], 8u);
                __builtin_memcpy(&regfile[16u + i * 2u],      &e[i][0], 8u);
                __builtin_memcpy(&regfile[16u + i * 2u + 1u], &e[i][1], 8u);
                __builtin_memcpy(&regfile[24u + i * 2u],      &a[i][0], 8u);
                __builtin_memcpy(&regfile[24u + i * 2u + 1u], &a[i][1], 8u);
            }
            uint64_t new_seed[8];
            rx_blake2b_512_regfile(regfile, new_seed);
            for (uint32_t i{ 0u }; i < 8u; ++i)
            {
                uint64_t const w{ new_seed[i] };
                gen4r_state[i * 8u + 0u] = (uint8_t)w;
                gen4r_state[i * 8u + 1u] = (uint8_t)(w >> 8u);
                gen4r_state[i * 8u + 2u] = (uint8_t)(w >> 16u);
                gen4r_state[i * 8u + 3u] = (uint8_t)(w >> 24u);
                gen4r_state[i * 8u + 4u] = (uint8_t)(w >> 32u);
                gen4r_state[i * 8u + 5u] = (uint8_t)(w >> 40u);
                gen4r_state[i * 8u + 6u] = (uint8_t)(w >> 48u);
                gen4r_state[i * 8u + 7u] = (uint8_t)(w >> 56u);
            }
        }
    }

    // ── Step 5: finalise ──────────────────────────────────────────────────
    uint64_t aes_hash[8];
    aes_hash1r(sp, aes_hash);

    uint64_t regfile[32]{};
    for (uint32_t i{ 0u }; i < 8u; ++i) { regfile[i] = r[i]; }
    for (uint32_t i{ 0u }; i < 4u; ++i)
    {
        __builtin_memcpy(&regfile[8u  + i * 2u],      &f[i][0], 8u);
        __builtin_memcpy(&regfile[8u  + i * 2u + 1u], &f[i][1], 8u);
        __builtin_memcpy(&regfile[16u + i * 2u],      &e[i][0], 8u);
        __builtin_memcpy(&regfile[16u + i * 2u + 1u], &e[i][1], 8u);
    }
    for (uint32_t i{ 0u }; i < 8u; ++i) { regfile[24u + i] = aes_hash[i]; }

    // Store final hash back into seed[0..3] (4 × uint64 = 32 bytes)
    rx_blake2b_256_regfile(regfile, seed);
}


///////////////////////////////////////////////////////////////////////////////
// Kernel: build cache (simplified Blake2b fill, not real Argon2d)
///////////////////////////////////////////////////////////////////////////////

__device__ __constant__
uint64_t RX_CACHE_IV[8]
{
    0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
    0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
    0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
    0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};

__device__ __constant__
uint8_t RX_CACHE_SIGMA[10][16]
{
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
};

__device__ __forceinline__
uint64_t rx_cache_ror64(uint64_t const x, uint32_t const n)
{
    return (x >> n) | (x << (64u - n));
}

__device__ __forceinline__
void rx_cache_b2b_G(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d,
                    uint64_t const x, uint64_t const y)
{
    a = a + b + x; d = rx_cache_ror64(d ^ a, 32u);
    c = c + d;     b = rx_cache_ror64(b ^ c, 24u);
    a = a + b + y; d = rx_cache_ror64(d ^ a, 16u);
    c = c + d;     b = rx_cache_ror64(b ^ c, 63u);
}

__device__ __forceinline__
void rx_cache_b2b_compress(uint64_t h[8], uint64_t const m[16],
                           uint64_t const t, uint32_t const last)
{
    uint64_t v[16];
    for (uint32_t i{ 0u }; i < 8u; ++i) { v[i] = h[i]; v[i + 8] = RX_CACHE_IV[i]; }
    v[12] ^= t;
    if (0u != last) { v[14] ^= 0xFFFFFFFFFFFFFFFFULL; }
    for (uint32_t r{ 0u }; r < 12u; ++r)
    {
        uint8_t const* const s{ RX_CACHE_SIGMA[r % 10u] };
        rx_cache_b2b_G(v[0], v[4], v[8],  v[12], m[s[0]],  m[s[1]]);
        rx_cache_b2b_G(v[1], v[5], v[9],  v[13], m[s[2]],  m[s[3]]);
        rx_cache_b2b_G(v[2], v[6], v[10], v[14], m[s[4]],  m[s[5]]);
        rx_cache_b2b_G(v[3], v[7], v[11], v[15], m[s[6]],  m[s[7]]);
        rx_cache_b2b_G(v[0], v[5], v[10], v[15], m[s[8]],  m[s[9]]);
        rx_cache_b2b_G(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        rx_cache_b2b_G(v[2], v[7], v[8],  v[13], m[s[12]], m[s[13]]);
        rx_cache_b2b_G(v[3], v[4], v[9],  v[14], m[s[14]], m[s[15]]);
    }
    for (uint32_t i{ 0u }; i < 8u; ++i) { h[i] ^= v[i] ^ v[i + 8u]; }
}

///////////////////////////////////////////////////////////////////////////////
// Dataset kernel: SuperscalarHash programs passed from CPU
///////////////////////////////////////////////////////////////////////////////

// GPU-friendly packed representation of 8 SuperscalarHash programs.
// Allocated in device global memory and passed to the dataset kernel.
struct GpuSuperscalarData
{
    uint32_t sizes  [algo::random_x::SUPERSCALAR_ITERS];
    uint32_t addregs[algo::random_x::SUPERSCALAR_ITERS];
    uint8_t  types  [algo::random_x::SUPERSCALAR_ITERS][algo::random_x::SUPERSCALAR_MAX_INSTRUCTIONS];
    uint8_t  dsts   [algo::random_x::SUPERSCALAR_ITERS][algo::random_x::SUPERSCALAR_MAX_INSTRUCTIONS];
    uint8_t  srcs   [algo::random_x::SUPERSCALAR_ITERS][algo::random_x::SUPERSCALAR_MAX_INSTRUCTIONS];
    uint32_t imms   [algo::random_x::SUPERSCALAR_ITERS][algo::random_x::SUPERSCALAR_MAX_INSTRUCTIONS];
    uint64_t rcps   [algo::random_x::SUPERSCALAR_ITERS][algo::random_x::SUPERSCALAR_MAX_INSTRUCTIONS];
};

// Dataset item register initialization constants (spec section 6.2)
static constexpr uint64_t RX_DS_INIT_MUL{ 6364136223846793005ULL };

__device__ __constant__
uint64_t RX_DS_XOR[7]
{
    9298411001130361340ULL,
    12065312585734608966ULL,
    9306329213124626780ULL,
    5281919268842080866ULL,
    10536153434571861004ULL,
    3398623926847679864ULL,
    9549104520008361294ULL,
};

__global__
void kernel_rx_build_dataset(
    uint64_t const* const          cache,
    uint64_t* const                dataset,
    GpuSuperscalarData const* const progData)
{
    uint64_t const stride{ static_cast<uint64_t>(gridDim.x) * blockDim.x };
    uint64_t       item  { static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x };

    while (item < RX_DATASET_ITEMS)
    {
        // Initialize r[0..7] from item index (spec section 6.2)
        uint64_t const r0{ (item + 1ull) * RX_DS_INIT_MUL };
        uint64_t r[8];
        r[0] = r0;
        for (uint32_t i{ 1u }; i < 8u; ++i) { r[i] = r0 ^ RX_DS_XOR[i - 1u]; }

        // 8 rounds: execute SuperscalarHash program then XOR with cache entry
        for (uint32_t p{ 0u }; p < algo::random_x::SUPERSCALAR_ITERS; ++p)
        {
            uint32_t const progSize{ progData->sizes[p] };

            for (uint32_t j{ 0u }; j < progSize; ++j)
            {
                uint8_t  const type{ progData->types[p][j] };
                uint8_t  const dst { progData->dsts [p][j] };
                uint8_t  const src { progData->srcs [p][j] };
                uint32_t const imm { progData->imms [p][j] };

                switch (type)
                {
                    case 0u:  // ISUB_R
                    {
                        r[dst] -= r[src];
                        break;
                    }
                    case 1u:  // IXOR_R
                    {
                        r[dst] ^= r[src];
                        break;
                    }
                    case 2u:  // IADD_RS: dst += src << (imm & 3)
                    {
                        r[dst] += r[src] << (imm & 3u);
                        break;
                    }
                    case 3u:  // IMUL_R
                    {
                        r[dst] *= r[src];
                        break;
                    }
                    case 4u:  // IROR_C: dst = ror(dst, imm & 63)
                    {
                        uint32_t const shift{ imm & 63u };
                        r[dst] = (r[dst] >> shift) | (r[dst] << (64u - shift));
                        break;
                    }
                    case 5u:  // IADD_C: dst += sign-extended imm32
                    {
                        r[dst] += static_cast<uint64_t>(static_cast<int32_t>(imm));
                        break;
                    }
                    case 6u:  // IXOR_C: dst ^= sign-extended imm32
                    {
                        r[dst] ^= static_cast<uint64_t>(static_cast<int32_t>(imm));
                        break;
                    }
                    case 7u:  // IMULH_R: dst = high64(dst * src) unsigned
                    {
                        r[dst] = __umul64hi(r[dst], r[src]);
                        break;
                    }
                    case 8u:  // ISMULH_R: dst = high64(dst * src) signed
                    {
                        r[dst] = static_cast<uint64_t>(
                            __mul64hi(static_cast<long long>(r[dst]),
                                      static_cast<long long>(r[src])));
                        break;
                    }
                    case 9u:  // IMUL_RCP: dst *= precomputed_reciprocal
                    {
                        uint64_t const rcp{ progData->rcps[p][j] };
                        if (0ull != rcp) { r[dst] *= rcp; }
                        break;
                    }
                    default: break;
                }
            }

            // XOR r[0..7] with cache entry indexed by addressReg
            uint64_t const        cacheIdx{ r[progData->addregs[p]] % RX_CACHE_ITEMS };
            uint64_t const* const cblk    { cache + cacheIdx * 8ull };
            for (uint32_t i{ 0u }; i < 8u; ++i) { r[i] ^= cblk[i]; }
        }

        uint64_t* const dest{ dataset + item * 8ull };
        for (uint32_t i{ 0u }; i < 8u; ++i) { dest[i] = r[i]; }

        item += stride;
    }
}


///////////////////////////////////////////////////////////////////////////////
// Kernel: search — 1 thread = 1 nonce
///////////////////////////////////////////////////////////////////////////////

__global__
void kernel_rx_search(
    uint64_t const* const         dataset,
    uint8_t* const                scratchpads,
    algo::random_x::Result* const resultCache)
{
    uint32_t const globalId{ blockIdx.x * blockDim.x + threadIdx.x };
    uint32_t const nonce   { static_cast<uint32_t>(rx_start_nonce + globalId) };

    // Build blob with nonce at offset 39
    uint8_t blob[RX_BLOB_SIZE];
    for (uint32_t i{ 0u }; i < RX_BLOB_SIZE; ++i) { blob[i] = rx_blob[i]; }
    blob[RX_BLOB_NONCE_OFFSET + 0u] = (uint8_t)(nonce);
    blob[RX_BLOB_NONCE_OFFSET + 1u] = (uint8_t)(nonce >> 8u);
    blob[RX_BLOB_NONCE_OFFSET + 2u] = (uint8_t)(nonce >> 16u);
    blob[RX_BLOB_NONCE_OFFSET + 3u] = (uint8_t)(nonce >> 24u);

    // Hash blob → 64-byte seed
    uint64_t seed[8];
    rx_blake2b_512_blob(blob, RX_BLOB_SIZE, seed);

    // Run VM — writes final 32-byte hash into seed[0..3]
    uint8_t* const sp{ scratchpads + (uint64_t)globalId * RX_SCRATCHPAD_STRIDE };
    rx_run_vm(seed, dataset, sp);

    // seed[0..3] now holds the 32-byte hash (Blake2b-256 output from rx_run_vm)
    // bytes 28-31 are the upper 32 bits of seed[3]
    uint32_t const hashLast4{ static_cast<uint32_t>(seed[3] >> 32u) };

    if (hashLast4 < rx_target)
    {
        if (0u == atomicCAS(&resultCache->count, 0u, 1u))
        {
            resultCache->nonces[0] = nonce;
            for (uint32_t i{ 0u }; i < 4u; ++i)
            {
                uint64_t const w{ seed[i] };
                resultCache->hash[i * 8u + 0u] = (uint8_t)w;
                resultCache->hash[i * 8u + 1u] = (uint8_t)(w >> 8u);
                resultCache->hash[i * 8u + 2u] = (uint8_t)(w >> 16u);
                resultCache->hash[i * 8u + 3u] = (uint8_t)(w >> 24u);
                resultCache->hash[i * 8u + 4u] = (uint8_t)(w >> 32u);
                resultCache->hash[i * 8u + 5u] = (uint8_t)(w >> 40u);
                resultCache->hash[i * 8u + 6u] = (uint8_t)(w >> 48u);
                resultCache->hash[i * 8u + 7u] = (uint8_t)(w >> 56u);
            }
            resultCache->found = true;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////
// Host wrappers
///////////////////////////////////////////////////////////////////////////////

__host__
bool randomxFreeMemory(resolver::nvidia::random_x::KernelParameters& params)
{
    CU_SAFE_DELETE(params.dataset);
    CU_SAFE_DELETE(params.scratchpads);
    CU_SAFE_DELETE_HOST(params.resultCache);
    return true;
}


__host__
bool randomxInitMemory(
    resolver::nvidia::random_x::KernelParameters& params,
    uint32_t const blocks,
    uint32_t const threads)
{
    constexpr uint64_t DATASET_BYTES   { RX_DATASET_ITEMS * 8ull * 8ull }; // items * 8 uint64 * 8 bytes
    uint64_t const     scratchpadsBytes{ static_cast<uint64_t>(blocks) * threads * RX_SCRATCHPAD_STRIDE };

    CU_ALLOC(&params.dataset,    DATASET_BYTES);
    CU_ALLOC(&params.scratchpads, scratchpadsBytes);
    CU_ALLOC_HOST(&params.resultCache, sizeof(algo::random_x::Result));

    return true;
}


__host__
bool randomxBuildCache(
    cudaStream_t   stream,
    uint8_t* const gpuCache,
    uint8_t const* seedHash)
{
    constexpr uint64_t CACHE_BYTES{ RX_CACHE_ITEMS * 64ull }; // 256 MiB

    // Build cache on CPU via Argon2d, then upload to GPU
    uint8_t* hostCache{ nullptr };
    CUDA_ER(cudaMallocHost(reinterpret_cast<void**>(&hostCache), CACHE_BYTES));

    algo::random_x::buildCache(hostCache, seedHash);

    CUDA_ER(cudaMemcpyAsync(gpuCache, hostCache, CACHE_BYTES, cudaMemcpyHostToDevice, stream));
    CUDA_ER(cudaStreamSynchronize(stream));

    CUDA_ER(cudaFreeHost(hostCache));

    return true;
}


__host__
bool randomxBuildDataset(
    cudaStream_t                                   stream,
    resolver::nvidia::random_x::KernelParameters&  params,
    uint8_t const*                                 gpuCache,
    uint8_t const*                                 seedHash)
{
    constexpr uint32_t BLOCKS { 4096u };
    constexpr uint32_t THREADS{ 256u };

    // Generate 8 SuperscalarHash programs on CPU
    algo::random_x::SuperscalarProgram programs[algo::random_x::SUPERSCALAR_ITERS];
    algo::random_x::buildSuperscalarPrograms(seedHash, programs);

    // Pack into GPU-friendly struct and precompute IMUL_RCP reciprocals
    GpuSuperscalarData* const hostData{ new GpuSuperscalarData{} };
    for (uint32_t p{ 0u }; p < algo::random_x::SUPERSCALAR_ITERS; ++p)
    {
        hostData->sizes  [p] = programs[p].size;
        hostData->addregs[p] = programs[p].addressReg;
        for (uint32_t j{ 0u }; j < programs[p].size; ++j)
        {
            algo::random_x::ScalarInst const& instr{ programs[p].instructions[j] };
            hostData->types[p][j] = static_cast<uint8_t>(instr.type);
            hostData->dsts [p][j] = instr.dst;
            hostData->srcs [p][j] = instr.src;
            hostData->imms [p][j] = instr.imm;
            if (instr.type == algo::random_x::ScalarInstType::IMUL_RCP)
            {
                hostData->rcps[p][j] = algo::random_x::superscalarComputeReciprocal(instr.imm);
            }
            else
            {
                hostData->rcps[p][j] = 0ull;
            }
        }
    }

    // Upload program data to device, then launch dataset kernel
    GpuSuperscalarData* devData{ nullptr };
    CUDA_ER(cudaMalloc(reinterpret_cast<void**>(&devData), sizeof(GpuSuperscalarData)));
    CUDA_ER(cudaMemcpyAsync(
        devData,
        hostData,
        sizeof(GpuSuperscalarData),
        cudaMemcpyHostToDevice,
        stream));

    kernel_rx_build_dataset<<<BLOCKS, THREADS, 0, stream>>>(
        reinterpret_cast<uint64_t const*>(gpuCache),
        params.dataset,
        devData);

    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    CUDA_ER(cudaFree(devData));
    delete hostData;

    return true;
}


__host__
bool randomxUpdateConstants(
    uint8_t const* const blob,
    uint32_t const       target,
    uint64_t const       startNonce)
{
    CUDA_ER(cudaMemcpyToSymbol(rx_blob,        blob,       RX_BLOB_SIZE));
    CUDA_ER(cudaMemcpyToSymbol(rx_target,      &target,    sizeof(uint32_t)));
    CUDA_ER(cudaMemcpyToSymbol(rx_start_nonce, &startNonce, sizeof(uint64_t)));

    return true;
}


__host__
bool randomxSearch(
    cudaStream_t stream,
    uint32_t const blocks,
    uint32_t const threads,
    resolver::nvidia::random_x::KernelParameters& params)
{
    params.resultCache->found = false;
    params.resultCache->count = 0u;

    kernel_rx_search<<<blocks, threads, 0, stream>>>(
        params.dataset,
        params.scratchpads,
        params.resultCache);
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
