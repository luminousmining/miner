#include <common/cuda/rotate_byte.cuh>


__constant__
uint8_t blake2b_sigma[12][16]
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
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13 , 0 },
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
};


__device__ __forceinline__
void blake2b_mix(
    uint64_t const* __restrict__ matrice,
    uint32_t const round,
    uint32_t const i,
    uint64_t& a,
    uint64_t& b,
    uint64_t& c,
    uint64_t& d)
{
    a += b + matrice[blake2b_sigma[round][i]];
    d = ror_64(d ^ a, 32);
    c += d;
    b = ror_64(b ^ c, 24);

    a += b + matrice[blake2b_sigma[round][i + 1]];
    d = ror_64(d ^ a, 16);
    c += d;
    b =  ror_64(b ^ c, 63);
}


__device__ __forceinline__
void blake2b_round(
    uint64_t const* __restrict__ matrice,
    uint64_t* __restrict__ v,
    uint64_t const r)
{
    blake2b_mix(matrice, r, 0,  v[0], v[4], v[8],  v[12]);
    blake2b_mix(matrice, r, 2,  v[1], v[5], v[9],  v[13]);
    blake2b_mix(matrice, r, 4,  v[2], v[6], v[10], v[14]);
    blake2b_mix(matrice, r, 6,  v[3], v[7], v[11], v[15]);
    blake2b_mix(matrice, r, 8,  v[0], v[5], v[10], v[15]);
    blake2b_mix(matrice, r, 10, v[1], v[6], v[11], v[12]);
    blake2b_mix(matrice, r, 12, v[2], v[7], v[8],  v[13]);
    blake2b_mix(matrice, r, 14, v[3], v[4], v[9],  v[14]);
}


__device__ __forceinline__
void blake2b(
    uint64_t* __restrict__ h,
    uint64_t const* __restrict__ matrice,
    uint64_t const t,
    uint64_t const f)
{
    uint64_t v[16];

    v[0] = h[0];
    v[1] = h[1];
    v[2] = h[2];
    v[3] = h[3];
    v[4] = h[4];
    v[5] = h[5];
    v[6] = h[6];
    v[7] = h[7];

    v[8]  = 0x6A09E667F3BCC908ul;
    v[9]  = 0xBB67AE8584CAA73Bul;
    v[10] = 0x3C6EF372FE94F82Bul;
    v[11] = 0xA54FF53A5F1D36F1ul;
    v[12] = 0x510E527FADE682D1ul;
    v[13] = 0x9B05688C2B3E6C1Ful;
    v[14] = 0x1F83D9ABFB41BD6Bul;
    v[15] = 0x5BE0CD19137E2179ul;

    v[12] ^= t;
    v[14] ^= f;

    #pragma unroll
    for (uint32_t round= 0u; round < 12u; ++round)
    {
        blake2b_round(matrice, v, round);
    }

    h[0] ^= v[0] ^ v[0 + 8];
    h[1] ^= v[1] ^ v[1 + 8];
    h[2] ^= v[2] ^ v[2 + 8];
    h[3] ^= v[3] ^ v[3 + 8];
    h[4] ^= v[4] ^ v[4 + 8];
    h[5] ^= v[5] ^ v[5 + 8];
    h[6] ^= v[6] ^ v[6 + 8];
    h[7] ^= v[7] ^ v[7 + 8];
}
