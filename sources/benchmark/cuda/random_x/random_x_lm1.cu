///////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////////
#include <common/error/cuda_error.hpp>


///////////////////////////////////////////////////////////////////////////////
// RandomX constants
///////////////////////////////////////////////////////////////////////////////
constexpr uint64_t RANDOMX_DATASET_ITEMS      { 34078720ull };
constexpr uint32_t RANDOMX_SCRATCHPAD_L3      { 2097152u };
constexpr uint32_t RANDOMX_SCRATCHPAD_L2      { 262144u };
constexpr uint32_t RANDOMX_SCRATCHPAD_L1      { 16384u };
constexpr uint32_t RANDOMX_PROGRAM_COUNT      { 8u };
constexpr uint32_t RANDOMX_PROGRAM_ITERATIONS { 2048u };
constexpr uint32_t RANDOMX_PROGRAM_SIZE       { 256u };
constexpr uint32_t RANDOMX_JUMP_OFFSET        { 8u };

// Scratchpad address masks
constexpr uint32_t MASK_L3_8  { 0x1FFFF8u };
constexpr uint32_t MASK_L2_8  { 0x03FFF8u };
constexpr uint32_t MASK_L1_8  { 0x003FF8u };
constexpr uint32_t MASK_L3_64 { 0x1FFFC0u };


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

// Blake2b-512: hash a 64-bit nonce → 64-byte output stored in h[8]
__device__
void rx_blake2b_512_nonce(uint64_t const nonce, uint64_t h[8])
{
    for (uint32_t i{ 0u }; i < 8u; ++i) { h[i] = RX_B2B_IV[i]; }
    h[0] ^= 0x0000000001010040ULL; // parameter block: digest=64, no key

    uint64_t m[16]{};
    m[0] = nonce;
    rx_b2b_compress(h, m, 8ull, 1u);
}

// Blake2b-256: hash 256 bytes (RegisterFile) → 32-byte output stored in out[4]
__device__
void rx_blake2b_256_regfile(uint64_t const* const regfile, uint64_t out[4])
{
    uint64_t h[8];
    for (uint32_t i{ 0u }; i < 8u; ++i) { h[i] = RX_B2B_IV[i]; }
    h[0] ^= 0x0000000001010020ULL; // digest=32, no key

    // Block 1 (bytes 0-127)
    uint64_t m[16];
    for (uint32_t i{ 0u }; i < 16u; ++i) { m[i] = regfile[i]; }
    rx_b2b_compress(h, m, 128ull, 0u);

    // Block 2 (bytes 128-255)
    for (uint32_t i{ 0u }; i < 16u; ++i) { m[i] = regfile[16u + i]; }
    rx_b2b_compress(h, m, 256ull, 1u);

    out[0] = h[0]; out[1] = h[1]; out[2] = h[2]; out[3] = h[3];
}

// Blake2b-512: hash 64 bytes (state rehash between programs)
__device__
void rx_blake2b_512_buf64(uint64_t const in[8], uint64_t out[8])
{
    for (uint32_t i{ 0u }; i < 8u; ++i) { out[i] = RX_B2B_IV[i]; }
    out[0] ^= 0x0000000001010040ULL;

    uint64_t m[16]{};
    for (uint32_t i{ 0u }; i < 8u; ++i) { m[i] = in[i]; }
    rx_b2b_compress(out, m, 64ull, 1u);
}


///////////////////////////////////////////////////////////////////////////////
// AES
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
    // AES state is column-major. ShiftRows rotates rows.
    uint8_t t;
    // Row 1: shift left 1
    t = s[1]; s[1] = s[5]; s[5] = s[9]; s[9] = s[13]; s[13] = t;
    // Row 2: shift left 2
    t = s[2]; s[2] = s[10]; s[10] = t;
    t = s[6]; s[6] = s[14]; s[14] = t;
    // Row 3: shift left 3 (= shift right 1)
    t = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = s[3]; s[3] = t;
}

__device__ __forceinline__
void aes_inv_shift_rows(uint8_t s[16])
{
    uint8_t t;
    // Row 1: shift right 1
    t = s[13]; s[13] = s[9]; s[9] = s[5]; s[5] = s[1]; s[1] = t;
    // Row 2: shift right 2
    t = s[2]; s[2] = s[10]; s[10] = t;
    t = s[6]; s[6] = s[14]; s[14] = t;
    // Row 3: shift right 3 (= shift left 1)
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
// AES Generator Keys (fixed constants from RandomX spec)
///////////////////////////////////////////////////////////////////////////////

// AesGenerator1R keys — derived from Blake2b_512("RandomX AesGenerator1R keys")
__device__ __constant__
uint8_t GEN1R_KEYS[4][16]
{
    { 0x53, 0xa5, 0xac, 0x6d, 0x09, 0x66, 0x71, 0x62, 0x2b, 0x55, 0xb5, 0xdb, 0x17, 0x49, 0xf4, 0xb4 },
    { 0x07, 0xaf, 0x7c, 0x6d, 0x0d, 0x71, 0x6a, 0x84, 0x78, 0xd3, 0x25, 0x17, 0x4e, 0xdc, 0xa1, 0x0d },
    { 0xf1, 0x62, 0x12, 0x3f, 0xc6, 0x7e, 0x94, 0x9f, 0x4f, 0x79, 0xc0, 0xf4, 0x45, 0xe3, 0x20, 0x3e },
    { 0x35, 0x81, 0xef, 0x6a, 0x7c, 0x31, 0xba, 0xb1, 0x88, 0x4c, 0x31, 0x16, 0x54, 0x91, 0x16, 0x49 }
};

// AesGenerator4R keys A — first set
__device__ __constant__
uint8_t GEN4R_KEYS_A[4][16]
{
    { 0xdd, 0xaa, 0x21, 0x64, 0xdb, 0x3d, 0x83, 0xd1, 0x2b, 0x6d, 0x54, 0x2f, 0x3f, 0xd2, 0xe5, 0x99 },
    { 0x50, 0x34, 0x0e, 0xb2, 0x55, 0x3f, 0x91, 0xb6, 0x53, 0x9d, 0xf7, 0x06, 0xe5, 0xcd, 0xdf, 0xa5 },
    { 0x04, 0xd9, 0x3e, 0x5c, 0xaf, 0x7b, 0x5e, 0x51, 0x9f, 0x67, 0xa4, 0x0a, 0xbf, 0x02, 0x1c, 0x17 },
    { 0x63, 0x37, 0x62, 0x85, 0x08, 0x5d, 0x8f, 0xe7, 0x85, 0x37, 0x67, 0xcd, 0x91, 0xd2, 0xde, 0xd8 }
};

// AesGenerator4R keys B — second set
__device__ __constant__
uint8_t GEN4R_KEYS_B[4][16]
{
    { 0x73, 0x6f, 0x82, 0xb5, 0xa6, 0xa7, 0xd6, 0xe3, 0x6d, 0x8b, 0x51, 0x3d, 0xb4, 0xff, 0x9e, 0x22 },
    { 0xf3, 0x6b, 0x56, 0xc7, 0xd9, 0xb3, 0x10, 0x9c, 0x4e, 0x4d, 0x02, 0xe9, 0xd2, 0xb7, 0x72, 0xb2 },
    { 0xe7, 0xc9, 0x73, 0xf2, 0x8b, 0xa3, 0x65, 0xf7, 0x0a, 0x66, 0xa9, 0x2b, 0xa7, 0xef, 0x3b, 0xf6 },
    { 0x09, 0xd6, 0x7c, 0x7a, 0xde, 0x39, 0x58, 0x91, 0xfd, 0xd1, 0x06, 0x0c, 0x2d, 0x76, 0xb0, 0xc0 }
};

// AesHash1R initial state
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
// AesGenerator1R — fill scratchpad 64 bytes at a time, 1 AES round per column
///////////////////////////////////////////////////////////////////////////////

__device__
void aes_gen1r_fill(uint8_t state[64], uint8_t* const dest, uint32_t const count)
{
    // state = 4 columns of 16 bytes
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


///////////////////////////////////////////////////////////////////////////////
// AesGenerator4R — generate program bytes, 4 AES rounds per column
///////////////////////////////////////////////////////////////////////////////

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


///////////////////////////////////////////////////////////////////////////////
// AesHash1R — scratchpad fingerprint
///////////////////////////////////////////////////////////////////////////////

__device__
void aes_hash1r(uint8_t const* const scratchpad, uint64_t out[8])
{
    uint8_t s0[16], s1[16], s2[16], s3[16];
    for (uint32_t i{ 0u }; i < 16u; ++i) { s0[i] = AESHASH1R_STATE[0][i]; }
    for (uint32_t i{ 0u }; i < 16u; ++i) { s1[i] = AESHASH1R_STATE[1][i]; }
    for (uint32_t i{ 0u }; i < 16u; ++i) { s2[i] = AESHASH1R_STATE[2][i]; }
    for (uint32_t i{ 0u }; i < 16u; ++i) { s3[i] = AESHASH1R_STATE[3][i]; }

    uint32_t const num_blocks{ RANDOMX_SCRATCHPAD_L3 / 64u };
    for (uint32_t blk{ 0u }; blk < num_blocks; ++blk)
    {
        uint8_t const* const k{ scratchpad + (uint64_t)blk * 64u };
        aes_enc_round(s0, k + 0u);
        aes_dec_round(s1, k + 16u);
        aes_enc_round(s2, k + 32u);
        aes_dec_round(s3, k + 48u);
    }

    // Two finalisation rounds
    aes_enc_round(s0, AESHASH1R_XKEYS[0]);
    aes_dec_round(s1, AESHASH1R_XKEYS[0]);
    aes_enc_round(s2, AESHASH1R_XKEYS[0]);
    aes_dec_round(s3, AESHASH1R_XKEYS[0]);

    aes_enc_round(s0, AESHASH1R_XKEYS[1]);
    aes_dec_round(s1, AESHASH1R_XKEYS[1]);
    aes_enc_round(s2, AESHASH1R_XKEYS[1]);
    aes_dec_round(s3, AESHASH1R_XKEYS[1]);

    // Write result as 8 uint64 (little-endian)
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        uint8_t const* const src{ (i < 2u) ? s0 : (i < 4u) ? s1 : (i < 6u) ? s2 : s3 };
        uint32_t const off{ (i % 2u) * 8u };
        out[i] = (uint64_t)src[off + 0u]        | ((uint64_t)src[off + 1u] << 8u)  |
                 ((uint64_t)src[off + 2u] << 16u) | ((uint64_t)src[off + 3u] << 24u) |
                 ((uint64_t)src[off + 4u] << 32u) | ((uint64_t)src[off + 5u] << 40u) |
                 ((uint64_t)src[off + 6u] << 48u) | ((uint64_t)src[off + 7u] << 56u);
    }
}


///////////////////////////////////////////////////////////////////////////////
// VM helpers
///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__
uint64_t sp_read_u64(uint8_t const* const sp, uint32_t const addr)
{
    uint8_t const* const p{ sp + addr };
    return (uint64_t)p[0]        | ((uint64_t)p[1] << 8u)  |
           ((uint64_t)p[2] << 16u) | ((uint64_t)p[3] << 24u) |
           ((uint64_t)p[4] << 32u) | ((uint64_t)p[5] << 40u) |
           ((uint64_t)p[6] << 48u) | ((uint64_t)p[7] << 56u);
}

__device__ __forceinline__
void sp_write_u64(uint8_t* const sp, uint32_t const addr, uint64_t const val)
{
    uint8_t* const p{ sp + addr };
    p[0] = (uint8_t)val;        p[1] = (uint8_t)(val >> 8u);
    p[2] = (uint8_t)(val >> 16u); p[3] = (uint8_t)(val >> 24u);
    p[4] = (uint8_t)(val >> 32u); p[5] = (uint8_t)(val >> 40u);
    p[6] = (uint8_t)(val >> 48u); p[7] = (uint8_t)(val >> 56u);
}

// Select scratchpad mask for _M instructions (read)
__device__ __forceinline__
uint32_t sp_mem_read_mask(uint8_t const src, uint8_t const dst, uint8_t const mod)
{
    if (src == dst) { return MASK_L3_8; }
    return (mod & 3u) ? MASK_L1_8 : MASK_L2_8;
}

// Select scratchpad mask for ISTORE (write)
__device__ __forceinline__
uint32_t sp_store_mask(uint8_t const mod)
{
    if ((mod >> 4u) >= 14u) { return MASK_L3_8; }
    return (mod & 3u) ? MASK_L1_8 : MASK_L2_8;
}

// Convert 4 scratchpad bytes (signed int32) to double (group F)
__device__ __forceinline__
double bytes_to_f_double(uint8_t const* const b)
{
    int32_t const i{ (int32_t)((uint32_t)b[0] | ((uint32_t)b[1] << 8u) |
                               ((uint32_t)b[2] << 16u) | ((uint32_t)b[3] << 24u)) };
    return (double)i;
}

// Convert 4 scratchpad bytes to double (group E) — guarantee positive, ≥ 1
__device__ __forceinline__
double bytes_to_e_double(uint8_t const* const b, uint64_t const eMask)
{
    int32_t const i{ (int32_t)((uint32_t)b[0] | ((uint32_t)b[1] << 8u) |
                               ((uint32_t)b[2] << 16u) | ((uint32_t)b[3] << 24u)) };
    double d{ (double)i };
    uint64_t bits;
    // Reinterpret and force positive, valid exponent
    __builtin_memcpy(&bits, &d, sizeof(bits));
    bits &= 0x000FFFFFE0000000ULL; // keep mantissa bits [51:29]
    bits |= eMask;                  // set sign=0, constrained exponent, low frac bits
    __builtin_memcpy(&d, &bits, sizeof(d));
    return d;
}

// Compute IMUL_RCP reciprocal: floor(2^64 / divisor)
__device__ __forceinline__
uint64_t rx_reciprocal(uint32_t const divisor)
{
    uint64_t const d{ (uint64_t)divisor };
    uint64_t const q{ UINT64_MAX / d };
    uint64_t const r{ UINT64_MAX - q * d };
    return (r + 1u >= d) ? q + 1u : q;
}


///////////////////////////////////////////////////////////////////////////////
// VM program execution
///////////////////////////////////////////////////////////////////////////////

struct RxProgram
{
    uint8_t  opcode[RANDOMX_PROGRAM_SIZE];
    uint8_t  dst[RANDOMX_PROGRAM_SIZE];
    uint8_t  src[RANDOMX_PROGRAM_SIZE];
    uint8_t  mod[RANDOMX_PROGRAM_SIZE];
    uint32_t imm32[RANDOMX_PROGRAM_SIZE];
};

__device__
void rx_parse_program(uint8_t const* const prog_bytes, RxProgram& prog)
{
    // Each instruction = 8 bytes: [imm32 LE 4B][mod 1B][src 1B][dst 1B][opcode 1B]
    for (uint32_t i{ 0u }; i < RANDOMX_PROGRAM_SIZE; ++i)
    {
        uint8_t const* const p{ prog_bytes + (uint64_t)i * 8u };
        prog.imm32[i]  = (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
                         ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
        prog.mod[i]    = p[4];
        prog.src[i]    = p[5];
        prog.dst[i]    = p[6];
        prog.opcode[i] = p[7];
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
    uint64_t const  eMask_hi)
{
    uint32_t last_mod[8];
    for (uint32_t i{ 0u }; i < 8u; ++i) { last_mod[i] = 0u; }

    uint32_t ip{ 0u };
    while (ip < RANDOMX_PROGRAM_SIZE)
    {
        uint8_t const  opc { prog.opcode[ip] };
        uint8_t const  di  { prog.dst[ip] };
        uint8_t const  si  { prog.src[ip] };
        uint8_t const  mod { prog.mod[ip] };
        uint32_t const imm { prog.imm32[ip] };
        int64_t  const imm_s{ (int64_t)(int32_t)imm };

        if (opc < 16u)
        {
            // IADD_RS
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint8_t const shift{ (mod >> 2u) & 3u };
            r[d] += r[s] << shift;
            if (d == 5u) { r[d] += (uint64_t)(int64_t)(int32_t)imm; }
            last_mod[d] = ip + 1u;
        }
        else if (opc < 23u)
        {
            // IADD_M
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const mask{ sp_mem_read_mask(s, d, mod) };
            uint32_t const addr{ static_cast<uint32_t>(r[s] + (uint64_t)imm_s) & mask };
            r[d] += sp_read_u64(sp, addr);
            last_mod[d] = ip + 1u;
        }
        else if (opc < 39u)
        {
            // ISUB_R
            uint8_t const d{ di & 7u }, s{ si & 7u };
            r[d] -= (d == s) ? (uint64_t)imm_s : r[s];
            last_mod[d] = ip + 1u;
        }
        else if (opc < 46u)
        {
            // ISUB_M
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const mask{ sp_mem_read_mask(s, d, mod) };
            uint32_t const addr{ static_cast<uint32_t>(r[s] + (uint64_t)imm_s) & mask };
            r[d] -= sp_read_u64(sp, addr);
            last_mod[d] = ip + 1u;
        }
        else if (opc < 62u)
        {
            // IMUL_R
            uint8_t const d{ di & 7u }, s{ si & 7u };
            r[d] *= (d == s) ? (uint64_t)imm_s : r[s];
            last_mod[d] = ip + 1u;
        }
        else if (opc < 66u)
        {
            // IMUL_M
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const mask{ sp_mem_read_mask(s, d, mod) };
            uint32_t const addr{ static_cast<uint32_t>(r[s] + (uint64_t)imm_s) & mask };
            r[d] *= sp_read_u64(sp, addr);
            last_mod[d] = ip + 1u;
        }
        else if (opc < 70u)
        {
            // IMULH_R
            uint8_t const d{ di & 7u }, s{ si & 7u };
            unsigned __int128 const prod{ (unsigned __int128)r[d] * (unsigned __int128)((d == s) ? r[d] : r[s]) };
            r[d] = (uint64_t)(prod >> 64u);
            last_mod[d] = ip + 1u;
        }
        else if (opc < 71u)
        {
            // IMULH_M
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const mask{ sp_mem_read_mask(s, d, mod) };
            uint32_t const addr{ static_cast<uint32_t>(r[s] + (uint64_t)imm_s) & mask };
            unsigned __int128 const prod{ (unsigned __int128)r[d] * (unsigned __int128)sp_read_u64(sp, addr) };
            r[d] = (uint64_t)(prod >> 64u);
            last_mod[d] = ip + 1u;
        }
        else if (opc < 75u)
        {
            // ISMULH_R
            uint8_t const d{ di & 7u }, s{ si & 7u };
            __int128 const prod{ (__int128)(int64_t)r[d] * (__int128)(int64_t)((d == s) ? r[d] : r[s]) };
            r[d] = (uint64_t)((unsigned __int128)prod >> 64u);
            last_mod[d] = ip + 1u;
        }
        else if (opc < 76u)
        {
            // ISMULH_M
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const mask{ sp_mem_read_mask(s, d, mod) };
            uint32_t const addr{ static_cast<uint32_t>(r[s] + (uint64_t)imm_s) & mask };
            __int128 const prod{ (__int128)(int64_t)r[d] * (__int128)(int64_t)sp_read_u64(sp, addr) };
            r[d] = (uint64_t)((unsigned __int128)prod >> 64u);
            last_mod[d] = ip + 1u;
        }
        else if (opc < 84u)
        {
            // IMUL_RCP
            uint8_t const d{ di & 7u };
            uint64_t const u{ (uint64_t)imm };
            bool const is_pow2{ (u != 0u) && ((u & (u - 1u)) == 0u) };
            if (u != 0u && !is_pow2)
            {
                r[d] *= rx_reciprocal(imm);
                last_mod[d] = ip + 1u;
            }
        }
        else if (opc < 86u)
        {
            // INEG_R
            uint8_t const d{ di & 7u };
            r[d] = ~r[d] + 1u;
            last_mod[d] = ip + 1u;
        }
        else if (opc < 101u)
        {
            // IXOR_R
            uint8_t const d{ di & 7u }, s{ si & 7u };
            r[d] ^= (d == s) ? (uint64_t)imm_s : r[s];
            last_mod[d] = ip + 1u;
        }
        else if (opc < 106u)
        {
            // IXOR_M
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const mask{ sp_mem_read_mask(s, d, mod) };
            uint32_t const addr{ static_cast<uint32_t>(r[s] + (uint64_t)imm_s) & mask };
            r[d] ^= sp_read_u64(sp, addr);
            last_mod[d] = ip + 1u;
        }
        else if (opc < 114u)
        {
            // IROR_R
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const n{ static_cast<uint32_t>((d == s) ? imm : r[s]) & 63u };
            r[d] = rx_ror64(r[d], n);
            last_mod[d] = ip + 1u;
        }
        else if (opc < 116u)
        {
            // IROL_R
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const n{ static_cast<uint32_t>((d == s) ? imm : r[s]) & 63u };
            r[d] = (r[d] << n) | (r[d] >> (64u - n));
            last_mod[d] = ip + 1u;
        }
        else if (opc < 120u)
        {
            // ISWAP_R
            uint8_t const d{ di & 7u }, s{ si & 7u };
            if (d != s)
            {
                uint64_t const tmp{ r[d] };
                r[d] = r[s]; r[s] = tmp;
                last_mod[d] = ip + 1u;
                last_mod[s] = ip + 1u;
            }
        }
        else if (opc < 124u)
        {
            // FSWAP_R — operates on F or E (dst & 7: 0-3 = F, 4-7 = E)
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
            // FADD_R — dst: F (di & 3), src: A (si & 3)
            uint8_t const d{ di & 3u }, s{ si & 3u };
            f[d][0] += a[s][0];
            f[d][1] += a[s][1];
        }
        else if (opc < 145u)
        {
            // FADD_M — dst: F (di & 3), src: R for address
            uint8_t const d{ di & 3u }, s{ si & 7u };
            uint32_t const addr{ static_cast<uint32_t>(r[s] + (uint64_t)imm_s) & MASK_L2_8 };
            uint8_t const* const mp{ sp + addr };
            f[d][0] += bytes_to_f_double(mp);
            f[d][1] += bytes_to_f_double(mp + 4u);
        }
        else if (opc < 161u)
        {
            // FSUB_R — dst: F (di & 3), src: A (si & 3)
            uint8_t const d{ di & 3u }, s{ si & 3u };
            f[d][0] -= a[s][0];
            f[d][1] -= a[s][1];
        }
        else if (opc < 166u)
        {
            // FSUB_M
            uint8_t const d{ di & 3u }, s{ si & 7u };
            uint32_t const addr{ static_cast<uint32_t>(r[s] + (uint64_t)imm_s) & MASK_L2_8 };
            uint8_t const* const mp{ sp + addr };
            f[d][0] -= bytes_to_f_double(mp);
            f[d][1] -= bytes_to_f_double(mp + 4u);
        }
        else if (opc < 172u)
        {
            // FSCAL_R — dst: F (di & 3), XOR IEEE bits with 0x80F0000000000000
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
            // FMUL_R — dst: E (di & 3), src: A (si & 3)
            uint8_t const d{ di & 3u }, s{ si & 3u };
            e[d][0] *= a[s][0];
            e[d][1] *= a[s][1];
        }
        else if (opc < 208u)
        {
            // FDIV_M — dst: E (di & 3), src: R for address
            uint8_t const d{ di & 3u }, s{ si & 7u };
            uint32_t const addr{ static_cast<uint32_t>(r[s] + (uint64_t)imm_s) & MASK_L2_8 };
            uint8_t const* const mp{ sp + addr };
            double const dlo{ bytes_to_e_double(mp,       eMask_lo) };
            double const dhi{ bytes_to_e_double(mp + 4u,  eMask_hi) };
            if (dlo != 0.0) { e[d][0] /= dlo; }
            if (dhi != 0.0) { e[d][1] /= dhi; }
        }
        else if (opc < 214u)
        {
            // FSQRT_R — dst: E (di & 3)
            uint8_t const d{ di & 3u };
            e[d][0] = sqrt(e[d][0]);
            e[d][1] = sqrt(e[d][1]);
        }
        else if (opc < 239u)
        {
            // CBRANCH
            uint8_t const d{ di & 7u };
            uint32_t const b { (mod >> 4u) + RANDOMX_JUMP_OFFSET }; // b in [8,23]
            int64_t cimm{ imm_s };
            cimm |= (int64_t)(1ull << b);
            if (b > 0u) { cimm &= ~(int64_t)(1ull << (b - 1u)); }

            uint32_t const target{ last_mod[d] };
            r[d] = (uint64_t)((int64_t)r[d] + cimm);
            for (uint32_t i{ 0u }; i < 8u; ++i) { last_mod[i] = ip + 1u; }

            if (((r[d] >> b) & 0xFFu) == 0u)
            {
                ip = target;
                continue; // skip ip++
            }
        }
        else if (opc < 240u)
        {
            // CFROUND
            uint8_t const s{ si & 7u };
            uint32_t const fprc{ static_cast<uint32_t>(rx_ror64(r[s], imm & 63u)) & 3u };
            // In lm1 brute, we do not change the hardware rounding mode (GPU limitation).
            // We store fprc but cannot apply it to double operations.
            (void)fprc;
        }
        else
        {
            // ISTORE — src: integer register, dst: address register
            uint8_t const d{ di & 7u }, s{ si & 7u };
            uint32_t const mask{ sp_store_mask(mod) };
            uint32_t const addr{ static_cast<uint32_t>(r[d] + (uint64_t)imm_s) & mask };
            sp_write_u64(sp, addr, r[s]);
        }

        ip++;
    }
}


///////////////////////////////////////////////////////////////////////////////
// Main kernel — 1 thread = 1 nonce, no collaboration
///////////////////////////////////////////////////////////////////////////////

__global__
void kernel_random_x_lm1(
    uint64_t const* const dataset,
    uint8_t* const        scratchpads)
{
    uint32_t const threadId{ blockIdx.x * blockDim.x + threadIdx.x };
    uint8_t* const sp{ scratchpads + (uint64_t)threadId * RANDOMX_SCRATCHPAD_L3 };

    // ── Step 1: generate seed from nonce ──────────────────────────────────
    uint64_t seed[8];
    rx_blake2b_512_nonce((uint64_t)threadId, seed);

    // ── Step 2: initialise scratchpad (2 MiB) via AesGenerator1R ─────────
    uint8_t aes_state[64];
    for (uint32_t i{ 0u }; i < 8u; ++i)
    {
        uint64_t const w{ seed[i] };
        uint32_t const off{ (i / 4u) * 32u + (i % 4u) * 8u };
        // Simplified: spread seed into AES state columns
        aes_state[(i % 4u) * 16u + 0u] = (uint8_t)w;
        aes_state[(i % 4u) * 16u + 1u] = (uint8_t)(w >> 8u);
        aes_state[(i % 4u) * 16u + 2u] = (uint8_t)(w >> 16u);
        aes_state[(i % 4u) * 16u + 3u] = (uint8_t)(w >> 24u);
        aes_state[(i % 4u) * 16u + 4u] = (uint8_t)(w >> 32u);
        aes_state[(i % 4u) * 16u + 5u] = (uint8_t)(w >> 40u);
        aes_state[(i % 4u) * 16u + 6u] = (uint8_t)(w >> 48u);
        aes_state[(i % 4u) * 16u + 7u] = (uint8_t)(w >> 56u);
        (void)off;
    }
    // Copy upper 32 bytes of seed into second half of AES state
    for (uint32_t i{ 4u }; i < 8u; ++i)
    {
        uint64_t const w{ seed[i] };
        uint32_t const col{ (i - 4u) * 16u };
        aes_state[col + 8u]  = (uint8_t)w;
        aes_state[col + 9u]  = (uint8_t)(w >> 8u);
        aes_state[col + 10u] = (uint8_t)(w >> 16u);
        aes_state[col + 11u] = (uint8_t)(w >> 24u);
        aes_state[col + 12u] = (uint8_t)(w >> 32u);
        aes_state[col + 13u] = (uint8_t)(w >> 40u);
        aes_state[col + 14u] = (uint8_t)(w >> 48u);
        aes_state[col + 15u] = (uint8_t)(w >> 56u);
    }

    // Fill scratchpad: 2097152 / 64 = 32768 calls to AesGenerator1R
    aes_gen1r_fill(aes_state, sp, RANDOMX_SCRATCHPAD_L3 / 64u);

    // ── Step 3: prepare AesGenerator4R state from end of Gen1R state ─────
    uint8_t gen4r_state[64];
    for (uint32_t i{ 0u }; i < 64u; ++i) { gen4r_state[i] = aes_state[i]; }

    // ── Step 4: VM loop — 8 programs × 2048 iterations ───────────────────

    // VM registers
    uint64_t r[8]{};
    double   f[4][2]{};
    double   e[4][2]{};
    double   a[4][2]{};

    // Pointer registers
    uint32_t ma{ 0u };
    uint32_t mx{ 0u };

    // Address read registers (set from program header, simplified defaults)
    uint8_t readReg[4]{ 0u, 2u, 4u, 6u };

    // E-register masks (set from program header, simplified defaults)
    uint64_t eMask_lo{ 0x3C00000000000000ULL }; // biased exp=0x3C0 → value ≈ 1
    uint64_t eMask_hi{ 0x3C00000000000000ULL };

    // Program buffer: AesGenerator4R produces 2176 bytes per program
    uint8_t prog_buf[2176];
    RxProgram prog{};

    for (uint32_t prog_idx{ 0u }; prog_idx < RANDOMX_PROGRAM_COUNT; ++prog_idx)
    {
        // Generate program bytes (34 × 64 bytes = 2176 bytes)
        aes_gen4r_fill(gen4r_state, prog_buf, 34u);

        // Parse program header (first 128 bytes = 16 × 8-byte quadwords)
        uint64_t hdr[16];
        for (uint32_t i{ 0u }; i < 16u; ++i)
        {
            uint8_t const* const p{ prog_buf + (uint64_t)i * 8u };
            hdr[i] = (uint64_t)p[0]        | ((uint64_t)p[1] << 8u)  |
                     ((uint64_t)p[2] << 16u) | ((uint64_t)p[3] << 24u) |
                     ((uint64_t)p[4] << 32u) | ((uint64_t)p[5] << 40u) |
                     ((uint64_t)p[6] << 48u) | ((uint64_t)p[7] << 56u);
        }

        // Init a0-a3 from header quadwords 0-7 (each qword → one double half)
        for (uint32_t i{ 0u }; i < 4u; ++i)
        {
            uint64_t const qlo{ hdr[i * 2u] };
            uint64_t const qhi{ hdr[i * 2u + 1u] };
            // Convert to IEEE-754 double in [1, 2^32)
            uint64_t const frac_lo{ qlo & 0x000FFFFFFFFFFFFFULL };
            uint32_t const exp_lo { static_cast<uint32_t>((qlo >> 59u) & 0x1Fu) };
            uint64_t const bits_lo{ ((uint64_t)(exp_lo + 1023u) << 52u) | frac_lo };
            uint64_t const frac_hi{ qhi & 0x000FFFFFFFFFFFFFULL };
            uint32_t const exp_hi { static_cast<uint32_t>((qhi >> 59u) & 0x1Fu) };
            uint64_t const bits_hi{ ((uint64_t)(exp_hi + 1023u) << 52u) | frac_hi };
            __builtin_memcpy(&a[i][0], &bits_lo, 8u);
            __builtin_memcpy(&a[i][1], &bits_hi, 8u);
        }

        // ma and mx from header quadwords 8 and 10
        ma = static_cast<uint32_t>(hdr[8]);
        mx = static_cast<uint32_t>(hdr[10]);

        // readReg from header quadword 12
        uint64_t const q12{ hdr[12] };
        readReg[0] = ((q12 >> 0u) & 1u) ? 1u : 0u;
        readReg[1] = ((q12 >> 1u) & 1u) ? 3u : 2u;
        readReg[2] = ((q12 >> 2u) & 1u) ? 5u : 4u;
        readReg[3] = ((q12 >> 3u) & 1u) ? 7u : 6u;

        // E masks from header quadwords 14 and 15
        uint64_t const q14{ hdr[14] };
        uint64_t const q15{ hdr[15] };
        eMask_lo = ((q14 >> 60u) << 55u) | (3ull << 52u) | (q14 & 0x3FFFFFull) | (1023ull << 52u);
        eMask_hi = ((q15 >> 60u) << 55u) | (3ull << 52u) | (q15 & 0x3FFFFFull) | (1023ull << 52u);

        // Parse 256 instructions from prog_buf[128..2175]
        rx_parse_program(prog_buf + 128u, prog);

        // ── VM execution: 2048 iterations ────────────────────────────────
        // Reset integer registers before loop
        for (uint32_t i{ 0u }; i < 8u; ++i) { r[i] = 0ull; }

        uint32_t spAddr0{ mx };
        uint32_t spAddr1{ ma };

        for (uint32_t iter{ 0u }; iter < RANDOMX_PROGRAM_ITERATIONS; ++iter)
        {
            // Step 1: update scratchpad address pointers
            uint64_t const rr01{ r[readReg[0]] ^ r[readReg[1]] };
            spAddr0 ^= static_cast<uint32_t>(rr01);
            spAddr1 ^= static_cast<uint32_t>(rr01 >> 32u);

            // Step 2: XOR r0-r7 with 64 bytes from Scratchpad[spAddr0 & L3_64]
            uint32_t const addr2{ spAddr0 & MASK_L3_64 };
            for (uint32_t i{ 0u }; i < 8u; ++i)
            {
                r[i] ^= sp_read_u64(sp, addr2 + i * 8u);
            }

            // Step 3: load f0-f3, e0-e3 from Scratchpad[spAddr1 & L3_64]
            uint32_t const addr3{ spAddr1 & MASK_L3_64 };
            uint8_t const* const blk3{ sp + addr3 };
            for (uint32_t i{ 0u }; i < 4u; ++i)
            {
                // Each f/e register gets its two halves from 8 bytes each
                uint8_t const* const lo{ blk3 + i * 8u };
                uint8_t const* const hi{ blk3 + 32u + i * 8u };
                f[i][0] = bytes_to_f_double(lo);
                f[i][1] = bytes_to_f_double(lo + 4u);
                e[i][0] = bytes_to_e_double(hi,       eMask_lo);
                e[i][1] = bytes_to_e_double(hi + 4u,  eMask_hi);
            }

            // Step 4: execute the 256-instruction program
            rx_execute_program(r, f, e, a, sp, prog, eMask_lo, eMask_hi);

            // Step 5: update mx
            uint32_t const rr23{ static_cast<uint32_t>(r[readReg[2]]) ^
                                  static_cast<uint32_t>(r[readReg[3]]) };
            mx ^= rr23;
            mx &= 0xFFFFFFC0u;

            // Step 6: read 64 bytes from dataset (full mode) at address ma
            uint64_t const        itemIdx{ (static_cast<uint64_t>(ma) >> 6u) % RANDOMX_DATASET_ITEMS };
            uint64_t const* const dblk   { dataset + itemIdx * 8ull };
            for (uint32_t i{ 0u }; i < 8u; ++i)
            {
                r[i] ^= dblk[i];
            }

            // Step 7: swap mx and ma
            { uint32_t const tmp{ mx }; mx = ma; ma = tmp; }

            // Step 8: write r0-r7 to scratchpad at spAddr1
            uint32_t const addr9{ spAddr1 & MASK_L3_64 };
            for (uint32_t i{ 0u }; i < 8u; ++i) { sp_write_u64(sp, addr9 + i * 8u, r[i]); }

            // Step 9: XOR f with e (bitwise on IEEE-754 bits)
            for (uint32_t i{ 0u }; i < 4u; ++i)
            {
                uint64_t fb0, fb1, eb0, eb1;
                __builtin_memcpy(&fb0, &f[i][0], 8u); __builtin_memcpy(&eb0, &e[i][0], 8u);
                __builtin_memcpy(&fb1, &f[i][1], 8u); __builtin_memcpy(&eb1, &e[i][1], 8u);
                fb0 ^= eb0; fb1 ^= eb1;
                __builtin_memcpy(&f[i][0], &fb0, 8u); __builtin_memcpy(&f[i][1], &fb1, 8u);
            }

            // Step 10: write f0-f3 to scratchpad at spAddr0
            uint32_t const addr11{ spAddr0 & MASK_L3_64 };
            for (uint32_t i{ 0u }; i < 4u; ++i)
            {
                uint64_t fb0, fb1;
                __builtin_memcpy(&fb0, &f[i][0], 8u);
                __builtin_memcpy(&fb1, &f[i][1], 8u);
                sp_write_u64(sp, addr11 + i * 8u,       fb0);
                sp_write_u64(sp, addr11 + i * 8u + 4u,  fb1);
            }

            // Step 11: reset address temporaries
            spAddr0 = 0u;
            spAddr1 = 0u;
        }

        // Reseed AesGenerator4R for next program (except after last)
        if (prog_idx < RANDOMX_PROGRAM_COUNT - 1u)
        {
            // RegisterFile = r0-r7 (64 bytes) + f0-f3 (64 bytes) + e0-e3 (64 bytes) + a0-a3 (64 bytes)
            uint64_t regfile[32]{};
            for (uint32_t i{ 0u }; i < 8u; ++i) { regfile[i] = r[i]; }
            for (uint32_t i{ 0u }; i < 4u; ++i)
            {
                __builtin_memcpy(&regfile[8u + i * 2u],       &f[i][0], 8u);
                __builtin_memcpy(&regfile[8u + i * 2u + 1u],  &f[i][1], 8u);
                __builtin_memcpy(&regfile[16u + i * 2u],      &e[i][0], 8u);
                __builtin_memcpy(&regfile[16u + i * 2u + 1u], &e[i][1], 8u);
                __builtin_memcpy(&regfile[24u + i * 2u],      &a[i][0], 8u);
                __builtin_memcpy(&regfile[24u + i * 2u + 1u], &a[i][1], 8u);
            }
            // Blake2b-512 of first 64 bytes of regfile → new gen4r seed
            uint64_t new_seed[8];
            rx_blake2b_512_buf64(regfile, new_seed);
            for (uint32_t i{ 0u }; i < 8u; ++i)
            {
                uint64_t const w{ new_seed[i] };
                gen4r_state[(i % 4u) * 16u + (i / 4u) * 8u + 0u] = (uint8_t)w;
                gen4r_state[(i % 4u) * 16u + (i / 4u) * 8u + 1u] = (uint8_t)(w >> 8u);
                gen4r_state[(i % 4u) * 16u + (i / 4u) * 8u + 2u] = (uint8_t)(w >> 16u);
                gen4r_state[(i % 4u) * 16u + (i / 4u) * 8u + 3u] = (uint8_t)(w >> 24u);
                gen4r_state[(i % 4u) * 16u + (i / 4u) * 8u + 4u] = (uint8_t)(w >> 32u);
                gen4r_state[(i % 4u) * 16u + (i / 4u) * 8u + 5u] = (uint8_t)(w >> 40u);
                gen4r_state[(i % 4u) * 16u + (i / 4u) * 8u + 6u] = (uint8_t)(w >> 48u);
                gen4r_state[(i % 4u) * 16u + (i / 4u) * 8u + 7u] = (uint8_t)(w >> 56u);
            }
        }
    }

    // ── Step 5: finalise ──────────────────────────────────────────────────

    // Scratchpad fingerprint via AesHash1R → 64 bytes
    uint64_t aes_hash[8];
    aes_hash1r(sp, aes_hash);

    // RegisterFile (256 bytes):
    // bytes   0.. 63 : r0-r7
    // bytes  64..127 : f0-f3
    // bytes 128..191 : e0-e3
    // bytes 192..255 : a0-a3 overwritten by aes_hash
    uint64_t regfile[32]{};
    for (uint32_t i{ 0u }; i < 8u; ++i) { regfile[i] = r[i]; }
    for (uint32_t i{ 0u }; i < 4u; ++i)
    {
        __builtin_memcpy(&regfile[8u + i * 2u],       &f[i][0], 8u);
        __builtin_memcpy(&regfile[8u + i * 2u + 1u],  &f[i][1], 8u);
        __builtin_memcpy(&regfile[16u + i * 2u],      &e[i][0], 8u);
        __builtin_memcpy(&regfile[16u + i * 2u + 1u], &e[i][1], 8u);
    }
    // Overwrite a0-a3 section (bytes 192-255) with AesHash1R output
    for (uint32_t i{ 0u }; i < 8u; ++i) { regfile[24u + i] = aes_hash[i]; }

    // Final hash — result discarded in benchmark
    uint64_t result[4];
    rx_blake2b_256_regfile(regfile, result);

    // Prevent compiler from eliminating the computation
    if (result[0] == 0xDEADBEEFDEADBEEFull && result[1] == 0xDEADBEEFDEADBEEFull)
    {
        sp[0] = 0xFFu;
    }
}


__host__
bool random_x_lm1(
    cudaStream_t        stream,
    uint64_t const*     dataset,
    uint8_t*            scratchpads,
    uint32_t const      blocks,
    uint32_t const      threads)
{
    kernel_random_x_lm1<<<blocks, threads, 0, stream>>>(dataset, scratchpads);
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
