#define B2B_IV(v)                                                              \
{                                                                              \
    ((ulong *)(v))[0] = 0x6A09E667F3BCC908;                                    \
    ((ulong *)(v))[1] = 0xBB67AE8584CAA73B;                                    \
    ((ulong *)(v))[2] = 0x3C6EF372FE94F82B;                                    \
    ((ulong *)(v))[3] = 0xA54FF53A5F1D36F1;                                    \
    ((ulong *)(v))[4] = 0x510E527FADE682D1;                                    \
    ((ulong *)(v))[5] = 0x9B05688C2B3E6C1F;                                    \
    ((ulong *)(v))[6] = 0x1F83D9ABFB41BD6B;                                    \
    ((ulong *)(v))[7] = 0x5BE0CD19137E2179;                                    \
}


// G mixing function
#define B2B_G(v, a, b, c, d, x, y)                                             \
{                                                                              \
    ((ulong *)(v))[a] += ((ulong *)(v))[b] + x;                                \
    ((ulong *)(v))[d] = ror_u64(((ulong *)(v))[d] ^ ((ulong *)(v))[a], 32);    \
    ((ulong *)(v))[c] += ((ulong *)(v))[d];                                    \
    ((ulong *)(v))[b] = ror_u64(((ulong *)(v))[b] ^ ((ulong *)(v))[c], 24);    \
    ((ulong *)(v))[a] += ((ulong *)(v))[b] + y;                                \
    ((ulong *)(v))[d] = ror_u64(((ulong *)(v))[d] ^ ((ulong *)(v))[a], 16);    \
    ((ulong *)(v))[c] += ((ulong *)(v))[d];                                    \
    ((ulong *)(v))[b] = ror_u64(((ulong *)(v))[b] ^ ((ulong *)(v))[c], 63);    \
}


// mixing rounds
#define B2B_MIX(v, m)                                                          \
{                                                                              \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[ 0], ((ulong *)(m))[ 1]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 2], ((ulong *)(m))[ 3]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 4], ((ulong *)(m))[ 5]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[ 6], ((ulong *)(m))[ 7]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 8], ((ulong *)(m))[ 9]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[10], ((ulong *)(m))[11]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[12], ((ulong *)(m))[13]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[14], ((ulong *)(m))[15]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[14], ((ulong *)(m))[10]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 4], ((ulong *)(m))[ 8]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 9], ((ulong *)(m))[15]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[13], ((ulong *)(m))[ 6]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 1], ((ulong *)(m))[12]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[ 0], ((ulong *)(m))[ 2]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[11], ((ulong *)(m))[ 7]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[ 5], ((ulong *)(m))[ 3]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[11], ((ulong *)(m))[ 8]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[12], ((ulong *)(m))[ 0]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 5], ((ulong *)(m))[ 2]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[15], ((ulong *)(m))[13]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[10], ((ulong *)(m))[14]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[ 3], ((ulong *)(m))[ 6]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[ 7], ((ulong *)(m))[ 1]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[ 9], ((ulong *)(m))[ 4]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[ 7], ((ulong *)(m))[ 9]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 3], ((ulong *)(m))[ 1]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[13], ((ulong *)(m))[12]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[11], ((ulong *)(m))[14]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 2], ((ulong *)(m))[ 6]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[ 5], ((ulong *)(m))[10]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[ 4], ((ulong *)(m))[ 0]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[15], ((ulong *)(m))[ 8]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[ 9], ((ulong *)(m))[ 0]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 5], ((ulong *)(m))[ 7]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 2], ((ulong *)(m))[ 4]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[10], ((ulong *)(m))[15]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[14], ((ulong *)(m))[ 1]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[11], ((ulong *)(m))[12]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[ 6], ((ulong *)(m))[ 8]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[ 3], ((ulong *)(m))[13]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[ 2], ((ulong *)(m))[12]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 6], ((ulong *)(m))[10]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 0], ((ulong *)(m))[11]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[ 8], ((ulong *)(m))[ 3]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 4], ((ulong *)(m))[13]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[ 7], ((ulong *)(m))[ 5]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[15], ((ulong *)(m))[14]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[ 1], ((ulong *)(m))[ 9]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[12], ((ulong *)(m))[ 5]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 1], ((ulong *)(m))[15]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[14], ((ulong *)(m))[13]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[ 4], ((ulong *)(m))[10]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 0], ((ulong *)(m))[ 7]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[ 6], ((ulong *)(m))[ 3]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[ 9], ((ulong *)(m))[ 2]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[ 8], ((ulong *)(m))[11]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[13], ((ulong *)(m))[11]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 7], ((ulong *)(m))[14]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[12], ((ulong *)(m))[ 1]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[ 3], ((ulong *)(m))[ 9]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 5], ((ulong *)(m))[ 0]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[15], ((ulong *)(m))[ 4]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[ 8], ((ulong *)(m))[ 6]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[ 2], ((ulong *)(m))[10]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[ 6], ((ulong *)(m))[15]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[14], ((ulong *)(m))[ 9]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[11], ((ulong *)(m))[ 3]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[ 0], ((ulong *)(m))[ 8]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[12], ((ulong *)(m))[ 2]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[13], ((ulong *)(m))[ 7]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[ 1], ((ulong *)(m))[ 4]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[10], ((ulong *)(m))[ 5]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[10], ((ulong *)(m))[ 2]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 8], ((ulong *)(m))[ 4]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 7], ((ulong *)(m))[ 6]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[ 1], ((ulong *)(m))[ 5]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[15], ((ulong *)(m))[11]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[ 9], ((ulong *)(m))[14]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[ 3], ((ulong *)(m))[12]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[13], ((ulong *)(m))[ 0]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[ 0], ((ulong *)(m))[ 1]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 2], ((ulong *)(m))[ 3]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 4], ((ulong *)(m))[ 5]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[ 6], ((ulong *)(m))[ 7]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 8], ((ulong *)(m))[ 9]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[10], ((ulong *)(m))[11]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[12], ((ulong *)(m))[13]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[14], ((ulong *)(m))[15]);            \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((ulong *)(m))[14], ((ulong *)(m))[10]);            \
    B2B_G(v, 1, 5,  9, 13, ((ulong *)(m))[ 4], ((ulong *)(m))[ 8]);            \
    B2B_G(v, 2, 6, 10, 14, ((ulong *)(m))[ 9], ((ulong *)(m))[15]);            \
    B2B_G(v, 3, 7, 11, 15, ((ulong *)(m))[13], ((ulong *)(m))[ 6]);            \
    B2B_G(v, 0, 5, 10, 15, ((ulong *)(m))[ 1], ((ulong *)(m))[12]);            \
    B2B_G(v, 1, 6, 11, 12, ((ulong *)(m))[ 0], ((ulong *)(m))[ 2]);            \
    B2B_G(v, 2, 7,  8, 13, ((ulong *)(m))[11], ((ulong *)(m))[ 7]);            \
    B2B_G(v, 3, 4,  9, 14, ((ulong *)(m))[ 5], ((ulong *)(m))[ 3]);            \
}
