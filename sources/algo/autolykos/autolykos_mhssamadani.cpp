#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/pem.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/hmac.h>
#include <openssl/opensslv.h>

#include <stdio.h>

#include <algo/autolykos/autolykos.hpp>
#include <common/log/log.hpp>


constexpr uint32_t BUF_SIZE_8{ 128u };
constexpr uint32_t NONCE_SIZE_8{ 8u };
constexpr uint32_t HEIGHT_SIZE{ 4u };
constexpr uint32_t NUM_SIZE_8{ 32u };
constexpr uint32_t PK_SIZE_8{ 33u };
constexpr uint32_t NUM_SIZE_4{ NUM_SIZE_8 << 1 };
constexpr uint32_t K_LEN{ 32u };
constexpr uint32_t CONST_MES_SIZE_8{ 8192u };
constexpr uint32_t INIT_N_LEN{ 0x4000000u };
constexpr uint32_t MAX_N_LEN{ 0x7FC9FF98u };
constexpr uint32_t IncreaseStart{ (600*1024) };
constexpr uint32_t IncreaseEnd{ (4198400) };
constexpr uint32_t IncreasePeriodForN{ (50*1024) };

#define ERROR_OPENSSL      "OpenSSL"

#define CALL(func, name)                                                       \
do                                                                             \
{                                                                              \
    if (!(func))                                                               \
    {                                                                          \
        fprintf(stderr, "ERROR:  " name " failed at %s: %d\n",__FILE__,__LINE__);\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}                                                                              \
while (0)


// BLAKE2b-256 hash state context
struct ctx_t
{
    uint8_t b[BUF_SIZE_8];
    uint64_t h[8];
    uint64_t t[2];
    uint32_t c;
};

// cyclic right rotation
#define ROTR64(x, y) (((x) >> (y)) ^ ((x) << (64 - (y))))

// G mixing function
#define B2B_G(v, a, b, c, d, x, y)                                             \
do                                                                             \
{                                                                              \
    ((uint64_t *)(v))[a] += ((uint64_t *)(v))[b] + x;                          \
    ((uint64_t *)(v))[d]                                                       \
        = ROTR64(((uint64_t *)(v))[d] ^ ((uint64_t *)(v))[a], 32);             \
    ((uint64_t *)(v))[c] += ((uint64_t *)(v))[d];                              \
    ((uint64_t *)(v))[b]                                                       \
        = ROTR64(((uint64_t *)(v))[b] ^ ((uint64_t *)(v))[c], 24);             \
    ((uint64_t *)(v))[a] += ((uint64_t *)(v))[b] + y;                          \
    ((uint64_t *)(v))[d]                                                       \
        = ROTR64(((uint64_t *)(v))[d] ^ ((uint64_t *)(v))[a], 16);             \
    ((uint64_t *)(v))[c] += ((uint64_t *)(v))[d];                              \
    ((uint64_t *)(v))[b]                                                       \
        = ROTR64(((uint64_t *)(v))[b] ^ ((uint64_t *)(v))[c], 63);             \
}                                                                              \
while (0)

// mixing rounds
#define B2B_MIX(v, m)                                                          \
do                                                                             \
{                                                                              \
    B2B_G(v, 0, 4,  8, 12, ((uint64_t *)(m))[ 0], ((uint64_t *)(m))[ 1]);      \
    B2B_G(v, 1, 5,  9, 13, ((uint64_t *)(m))[ 2], ((uint64_t *)(m))[ 3]);      \
    B2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[ 4], ((uint64_t *)(m))[ 5]);      \
    B2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[ 6], ((uint64_t *)(m))[ 7]);      \
    B2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[ 8], ((uint64_t *)(m))[ 9]);      \
    B2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[10], ((uint64_t *)(m))[11]);      \
    B2B_G(v, 2, 7,  8, 13, ((uint64_t *)(m))[12], ((uint64_t *)(m))[13]);      \
    B2B_G(v, 3, 4,  9, 14, ((uint64_t *)(m))[14], ((uint64_t *)(m))[15]);      \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((uint64_t *)(m))[14], ((uint64_t *)(m))[10]);      \
    B2B_G(v, 1, 5,  9, 13, ((uint64_t *)(m))[ 4], ((uint64_t *)(m))[ 8]);      \
    B2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[ 9], ((uint64_t *)(m))[15]);      \
    B2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[13], ((uint64_t *)(m))[ 6]);      \
    B2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[ 1], ((uint64_t *)(m))[12]);      \
    B2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[ 0], ((uint64_t *)(m))[ 2]);      \
    B2B_G(v, 2, 7,  8, 13, ((uint64_t *)(m))[11], ((uint64_t *)(m))[ 7]);      \
    B2B_G(v, 3, 4,  9, 14, ((uint64_t *)(m))[ 5], ((uint64_t *)(m))[ 3]);      \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((uint64_t *)(m))[11], ((uint64_t *)(m))[ 8]);      \
    B2B_G(v, 1, 5,  9, 13, ((uint64_t *)(m))[12], ((uint64_t *)(m))[ 0]);      \
    B2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[ 5], ((uint64_t *)(m))[ 2]);      \
    B2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[15], ((uint64_t *)(m))[13]);      \
    B2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[10], ((uint64_t *)(m))[14]);      \
    B2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[ 3], ((uint64_t *)(m))[ 6]);      \
    B2B_G(v, 2, 7,  8, 13, ((uint64_t *)(m))[ 7], ((uint64_t *)(m))[ 1]);      \
    B2B_G(v, 3, 4,  9, 14, ((uint64_t *)(m))[ 9], ((uint64_t *)(m))[ 4]);      \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((uint64_t *)(m))[ 7], ((uint64_t *)(m))[ 9]);      \
    B2B_G(v, 1, 5,  9, 13, ((uint64_t *)(m))[ 3], ((uint64_t *)(m))[ 1]);      \
    B2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[13], ((uint64_t *)(m))[12]);      \
    B2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[11], ((uint64_t *)(m))[14]);      \
    B2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[ 2], ((uint64_t *)(m))[ 6]);      \
    B2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[ 5], ((uint64_t *)(m))[10]);      \
    B2B_G(v, 2, 7,  8, 13, ((uint64_t *)(m))[ 4], ((uint64_t *)(m))[ 0]);      \
    B2B_G(v, 3, 4,  9, 14, ((uint64_t *)(m))[15], ((uint64_t *)(m))[ 8]);      \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((uint64_t *)(m))[ 9], ((uint64_t *)(m))[ 0]);      \
    B2B_G(v, 1, 5,  9, 13, ((uint64_t *)(m))[ 5], ((uint64_t *)(m))[ 7]);      \
    B2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[ 2], ((uint64_t *)(m))[ 4]);      \
    B2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[10], ((uint64_t *)(m))[15]);      \
    B2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[14], ((uint64_t *)(m))[ 1]);      \
    B2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[11], ((uint64_t *)(m))[12]);      \
    B2B_G(v, 2, 7,  8, 13, ((uint64_t *)(m))[ 6], ((uint64_t *)(m))[ 8]);      \
    B2B_G(v, 3, 4,  9, 14, ((uint64_t *)(m))[ 3], ((uint64_t *)(m))[13]);      \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((uint64_t *)(m))[ 2], ((uint64_t *)(m))[12]);      \
    B2B_G(v, 1, 5,  9, 13, ((uint64_t *)(m))[ 6], ((uint64_t *)(m))[10]);      \
    B2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[ 0], ((uint64_t *)(m))[11]);      \
    B2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[ 8], ((uint64_t *)(m))[ 3]);      \
    B2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[ 4], ((uint64_t *)(m))[13]);      \
    B2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[ 7], ((uint64_t *)(m))[ 5]);      \
    B2B_G(v, 2, 7,  8, 13, ((uint64_t *)(m))[15], ((uint64_t *)(m))[14]);      \
    B2B_G(v, 3, 4,  9, 14, ((uint64_t *)(m))[ 1], ((uint64_t *)(m))[ 9]);      \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((uint64_t *)(m))[12], ((uint64_t *)(m))[ 5]);      \
    B2B_G(v, 1, 5,  9, 13, ((uint64_t *)(m))[ 1], ((uint64_t *)(m))[15]);      \
    B2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[14], ((uint64_t *)(m))[13]);      \
    B2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[ 4], ((uint64_t *)(m))[10]);      \
    B2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[ 0], ((uint64_t *)(m))[ 7]);      \
    B2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[ 6], ((uint64_t *)(m))[ 3]);      \
    B2B_G(v, 2, 7,  8, 13, ((uint64_t *)(m))[ 9], ((uint64_t *)(m))[ 2]);      \
    B2B_G(v, 3, 4,  9, 14, ((uint64_t *)(m))[ 8], ((uint64_t *)(m))[11]);      \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((uint64_t *)(m))[13], ((uint64_t *)(m))[11]);      \
    B2B_G(v, 1, 5,  9, 13, ((uint64_t *)(m))[ 7], ((uint64_t *)(m))[14]);      \
    B2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[12], ((uint64_t *)(m))[ 1]);      \
    B2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[ 3], ((uint64_t *)(m))[ 9]);      \
    B2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[ 5], ((uint64_t *)(m))[ 0]);      \
    B2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[15], ((uint64_t *)(m))[ 4]);      \
    B2B_G(v, 2, 7,  8, 13, ((uint64_t *)(m))[ 8], ((uint64_t *)(m))[ 6]);      \
    B2B_G(v, 3, 4,  9, 14, ((uint64_t *)(m))[ 2], ((uint64_t *)(m))[10]);      \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((uint64_t *)(m))[ 6], ((uint64_t *)(m))[15]);      \
    B2B_G(v, 1, 5,  9, 13, ((uint64_t *)(m))[14], ((uint64_t *)(m))[ 9]);      \
    B2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[11], ((uint64_t *)(m))[ 3]);      \
    B2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[ 0], ((uint64_t *)(m))[ 8]);      \
    B2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[12], ((uint64_t *)(m))[ 2]);      \
    B2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[13], ((uint64_t *)(m))[ 7]);      \
    B2B_G(v, 2, 7,  8, 13, ((uint64_t *)(m))[ 1], ((uint64_t *)(m))[ 4]);      \
    B2B_G(v, 3, 4,  9, 14, ((uint64_t *)(m))[10], ((uint64_t *)(m))[ 5]);      \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((uint64_t *)(m))[10], ((uint64_t *)(m))[ 2]);      \
    B2B_G(v, 1, 5,  9, 13, ((uint64_t *)(m))[ 8], ((uint64_t *)(m))[ 4]);      \
    B2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[ 7], ((uint64_t *)(m))[ 6]);      \
    B2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[ 1], ((uint64_t *)(m))[ 5]);      \
    B2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[15], ((uint64_t *)(m))[11]);      \
    B2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[ 9], ((uint64_t *)(m))[14]);      \
    B2B_G(v, 2, 7,  8, 13, ((uint64_t *)(m))[ 3], ((uint64_t *)(m))[12]);      \
    B2B_G(v, 3, 4,  9, 14, ((uint64_t *)(m))[13], ((uint64_t *)(m))[ 0]);      \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((uint64_t *)(m))[ 0], ((uint64_t *)(m))[ 1]);      \
    B2B_G(v, 1, 5,  9, 13, ((uint64_t *)(m))[ 2], ((uint64_t *)(m))[ 3]);      \
    B2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[ 4], ((uint64_t *)(m))[ 5]);      \
    B2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[ 6], ((uint64_t *)(m))[ 7]);      \
    B2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[ 8], ((uint64_t *)(m))[ 9]);      \
    B2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[10], ((uint64_t *)(m))[11]);      \
    B2B_G(v, 2, 7,  8, 13, ((uint64_t *)(m))[12], ((uint64_t *)(m))[13]);      \
    B2B_G(v, 3, 4,  9, 14, ((uint64_t *)(m))[14], ((uint64_t *)(m))[15]);      \
                                                                               \
    B2B_G(v, 0, 4,  8, 12, ((uint64_t *)(m))[14], ((uint64_t *)(m))[10]);      \
    B2B_G(v, 1, 5,  9, 13, ((uint64_t *)(m))[ 4], ((uint64_t *)(m))[ 8]);      \
    B2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[ 9], ((uint64_t *)(m))[15]);      \
    B2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[13], ((uint64_t *)(m))[ 6]);      \
    B2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[ 1], ((uint64_t *)(m))[12]);      \
    B2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[ 0], ((uint64_t *)(m))[ 2]);      \
    B2B_G(v, 2, 7,  8, 13, ((uint64_t *)(m))[11], ((uint64_t *)(m))[ 7]);      \
    B2B_G(v, 3, 4,  9, 14, ((uint64_t *)(m))[ 5], ((uint64_t *)(m))[ 3]);      \
}                                                                              \
while (0)

// initialization vector
#define B2B_IV(v)                                                              \
do                                                                             \
{                                                                              \
    ((uint64_t *)(v))[0] = 0x6A09E667F3BCC908;                                 \
    ((uint64_t *)(v))[1] = 0xBB67AE8584CAA73B;                                 \
    ((uint64_t *)(v))[2] = 0x3C6EF372FE94F82B;                                 \
    ((uint64_t *)(v))[3] = 0xA54FF53A5F1D36F1;                                 \
    ((uint64_t *)(v))[4] = 0x510E527FADE682D1;                                 \
    ((uint64_t *)(v))[5] = 0x9B05688C2B3E6C1F;                                 \
    ((uint64_t *)(v))[6] = 0x1F83D9ABFB41BD6B;                                 \
    ((uint64_t *)(v))[7] = 0x5BE0CD19137E2179;                                 \
}                                                                              \
while (0)

// blake2b initialization
#define B2B_INIT(ctx, aux)                                                     \
do                                                                             \
{                                                                              \
    ((uint64_t *)(aux))[0] = ((ctx_t *)(ctx))->h[0];                           \
    ((uint64_t *)(aux))[1] = ((ctx_t *)(ctx))->h[1];                           \
    ((uint64_t *)(aux))[2] = ((ctx_t *)(ctx))->h[2];                           \
    ((uint64_t *)(aux))[3] = ((ctx_t *)(ctx))->h[3];                           \
    ((uint64_t *)(aux))[4] = ((ctx_t *)(ctx))->h[4];                           \
    ((uint64_t *)(aux))[5] = ((ctx_t *)(ctx))->h[5];                           \
    ((uint64_t *)(aux))[6] = ((ctx_t *)(ctx))->h[6];                           \
    ((uint64_t *)(aux))[7] = ((ctx_t *)(ctx))->h[7];                           \
                                                                               \
    B2B_IV(aux + 8);                                                           \
                                                                               \
    ((uint64_t *)(aux))[12] ^= ((ctx_t *)(ctx))->t[0];                         \
    ((uint64_t *)(aux))[13] ^= ((ctx_t *)(ctx))->t[1];                         \
}                                                                              \
while (0)

// blake2b mixing
#define B2B_FINAL(ctx, aux)                                                    \
do                                                                             \
{                                                                              \
    ((uint64_t *)(aux))[16] = ((uint64_t *)(((ctx_t *)(ctx))->b))[ 0];         \
    ((uint64_t *)(aux))[17] = ((uint64_t *)(((ctx_t *)(ctx))->b))[ 1];         \
    ((uint64_t *)(aux))[18] = ((uint64_t *)(((ctx_t *)(ctx))->b))[ 2];         \
    ((uint64_t *)(aux))[19] = ((uint64_t *)(((ctx_t *)(ctx))->b))[ 3];         \
    ((uint64_t *)(aux))[20] = ((uint64_t *)(((ctx_t *)(ctx))->b))[ 4];         \
    ((uint64_t *)(aux))[21] = ((uint64_t *)(((ctx_t *)(ctx))->b))[ 5];         \
    ((uint64_t *)(aux))[22] = ((uint64_t *)(((ctx_t *)(ctx))->b))[ 6];         \
    ((uint64_t *)(aux))[23] = ((uint64_t *)(((ctx_t *)(ctx))->b))[ 7];         \
    ((uint64_t *)(aux))[24] = ((uint64_t *)(((ctx_t *)(ctx))->b))[ 8];         \
    ((uint64_t *)(aux))[25] = ((uint64_t *)(((ctx_t *)(ctx))->b))[ 9];         \
    ((uint64_t *)(aux))[26] = ((uint64_t *)(((ctx_t *)(ctx))->b))[10];         \
    ((uint64_t *)(aux))[27] = ((uint64_t *)(((ctx_t *)(ctx))->b))[11];         \
    ((uint64_t *)(aux))[28] = ((uint64_t *)(((ctx_t *)(ctx))->b))[12];         \
    ((uint64_t *)(aux))[29] = ((uint64_t *)(((ctx_t *)(ctx))->b))[13];         \
    ((uint64_t *)(aux))[30] = ((uint64_t *)(((ctx_t *)(ctx))->b))[14];         \
    ((uint64_t *)(aux))[31] = ((uint64_t *)(((ctx_t *)(ctx))->b))[15];         \
                                                                               \
    B2B_MIX(aux, aux + 16);                                                    \
                                                                               \
    ((ctx_t *)(ctx))->h[0] ^= ((uint64_t *)(aux))[0] ^ ((uint64_t *)(aux))[ 8];\
    ((ctx_t *)(ctx))->h[1] ^= ((uint64_t *)(aux))[1] ^ ((uint64_t *)(aux))[ 9];\
    ((ctx_t *)(ctx))->h[2] ^= ((uint64_t *)(aux))[2] ^ ((uint64_t *)(aux))[10];\
    ((ctx_t *)(ctx))->h[3] ^= ((uint64_t *)(aux))[3] ^ ((uint64_t *)(aux))[11];\
    ((ctx_t *)(ctx))->h[4] ^= ((uint64_t *)(aux))[4] ^ ((uint64_t *)(aux))[12];\
    ((ctx_t *)(ctx))->h[5] ^= ((uint64_t *)(aux))[5] ^ ((uint64_t *)(aux))[13];\
    ((ctx_t *)(ctx))->h[6] ^= ((uint64_t *)(aux))[6] ^ ((uint64_t *)(aux))[14];\
    ((ctx_t *)(ctx))->h[7] ^= ((uint64_t *)(aux))[7] ^ ((uint64_t *)(aux))[15];\
}                                                                              \
while (0)

// blake2b intermediate mixing procedure on host
#define HOST_B2B_H(ctx, aux)                                                   \
do                                                                             \
{                                                                              \
    ((ctx_t *)(ctx))->t[0] += BUF_SIZE_8;                                      \
    ((ctx_t *)(ctx))->t[1] += 1 - !(((ctx_t *)(ctx))->t[0] < BUF_SIZE_8);      \
                                                                               \
    B2B_INIT(ctx, aux);                                                        \
    B2B_FINAL(ctx, aux);                                                       \
                                                                               \
    ((ctx_t *)(ctx))->c = 0;                                                   \
}                                                                              \
while (0)

// blake2b intermediate mixing procedure on host
#define HOST_B2B_H_LAST(ctx, aux)                                              \
do                                                                             \
{                                                                              \
    ((ctx_t *)(ctx))->t[0] += ((ctx_t *)(ctx))->c;                             \
    ((ctx_t *)(ctx))->t[1]                                                     \
        += 1 - !(((ctx_t *)(ctx))->t[0] < ((ctx_t *)(ctx))->c);                \
                                                                               \
    while (((ctx_t *)(ctx))->c < BUF_SIZE_8)                                   \
    {                                                                          \
        ((ctx_t *)(ctx))->b[((ctx_t *)(ctx))->c++] = 0;                        \
    }                                                                          \
                                                                               \
    B2B_INIT(ctx, aux);                                                        \
                                                                               \
    ((uint64_t *)(aux))[14] = ~((uint64_t *)(aux))[14];                        \
                                                                               \
    B2B_FINAL(ctx, aux);                                                       \
}                                                                              \
while (0)


static uint32_t calcN(uint32_t const Hblock)
{
    uint32_t headerHeight;
    ((uint8_t *)&headerHeight)[0] = ((uint8_t *)&Hblock)[3];
    ((uint8_t *)&headerHeight)[1] = ((uint8_t *)&Hblock)[2];
    ((uint8_t *)&headerHeight)[2] = ((uint8_t *)&Hblock)[1];
    ((uint8_t *)&headerHeight)[3] = ((uint8_t *)&Hblock)[0];

    uint32_t newN = INIT_N_LEN;
    if (headerHeight < IncreaseStart)
    {
        newN = INIT_N_LEN;
    }
    else if (headerHeight >= IncreaseEnd)
    {
        newN = MAX_N_LEN;
    }
    else
    {
        uint32_t itersNumber = (headerHeight - IncreaseStart) / IncreasePeriodForN + 1;
        for (uint32_t i = 0; i < itersNumber; i++)
        {
            newN = newN / 100 * 105;
        }
    }
    return newN;
}

static void LittleEndianToHexStr(
    uint8_t const* in,
    uint32_t const inlen,
    char* out)
{
    uint8_t dig;

    for (int i = (inlen << 1) - 1; i >= 0; --i)
    {
        dig = (uint8_t)(in[i >> 1] >> ((i & 1) << 2)) & 0xF;

        out[(inlen << 1) - i - 1]
            = (dig <= 9)? (char)dig + '0': (char)dig + 'A' - 0xA;
    }

    out[inlen << 1] = '\0';

    return;
}


static void BigEndianToHexStr(
    const uint8_t * in,
    const uint32_t inlen,
    char * out)
{
    uint8_t dig;

    for (uint8_t i = 0; i < inlen << 1; ++i)
    {
        dig = (uint8_t)(in[i >> 1] >> (!(i & 1) << 2)) & 0xF;
        out[i] = (dig <= 9)? (char)dig + '0': (char)dig + 'A' - 0xA;
    }

    out[inlen << 1] = '\0';

    return;
}


static void HexStrToBigEndian(
    const char * in,
    const uint32_t inlen,
    uint8_t * out,
    const uint32_t outlen)
{
    memset(out, 0, outlen);

    for (uint8_t i = (outlen << 1) - inlen; i < (outlen << 1); ++i)
    {
        out[i >> 1]
            |= (((in[i] >= 'A')?  in[i] - 'A' + 0xA: in[i] - '0') & 0xF)
            << ((!(i & 1)) << 2);
    }

    return;
}


static void Blake2b256(
    const char * in,
    const int len,
    uint8_t * output,
    char * outstr)
{
    ctx_t ctx;
    uint64_t aux[32];

    //====================================================================//
    //  Initialize context
    //====================================================================//
    memset(ctx.b, 0, 128);
    B2B_IV(ctx.h);
    ctx.h[0] ^= 0x01010000 ^ NUM_SIZE_8;
    memset(ctx.t, 0, 16);
    ctx.c = 0;

    //====================================================================//
    //  Hash message
    //====================================================================//
    for (int i = 0; i < len; ++i)
    {
        if (ctx.c == 128) { HOST_B2B_H(&ctx, aux); }

        ctx.b[ctx.c++] = (uint8_t)(in[i]);
    }
    HOST_B2B_H_LAST(&ctx, aux);
    for (uint32_t i = 0u; i < NUM_SIZE_8; ++i)
    {
        output[NUM_SIZE_8 - i - 1] = (ctx.h[i >> 3] >> ((i & 7) << 3)) & 0xFF;
    }

    LittleEndianToHexStr(output, NUM_SIZE_8, outstr);
}


static void hashFn(
    const char * in,
    const int len,
    uint8_t * output,
    bool print = false)
{
    char *skstr = new char[len * 3];
    Blake2b256(in, len, output, skstr);

    uint8_t beHash[PK_SIZE_8];
    HexStrToBigEndian(skstr, NUM_SIZE_4, beHash, NUM_SIZE_8);

    memcpy(output, beHash, NUM_SIZE_8);

    delete[] skstr;
}


static void GenIdex(
    const char* in,
    const int len,
    uint32_t* index,
    uint64_t N_LEN)
{
    uint8_t sk[NUM_SIZE_8 * 2];
    char skstr[NUM_SIZE_4 + 10];

    memset(sk, 0, NUM_SIZE_8 * 2);
    memset(skstr, 0, NUM_SIZE_4);

    Blake2b256(in, len, sk, skstr);

    uint8_t beH[PK_SIZE_8];
    HexStrToBigEndian(skstr, NUM_SIZE_4, beH, NUM_SIZE_8);

    uint32_t* ind = index;

    memcpy(sk, beH, NUM_SIZE_8);
    memcpy(sk + NUM_SIZE_8, beH, NUM_SIZE_8);

    uint32_t tmpInd[32];
    int sliceIndex = 0;
    for (int k = 0; k < K_LEN; k++)
    {
        uint8_t tmp[4];
        memcpy(tmp, sk + sliceIndex, 4);
        memcpy(&tmpInd[k], sk + sliceIndex, 4);
        uint8_t tmp2[4];
        tmp2[0] = tmp[3];
        tmp2[1] = tmp[2];
        tmp2[2] = tmp[1];
        tmp2[3] = tmp[0];
        memcpy(&ind[k], tmp2, 4);
        ind[k] = ind[k] % N_LEN;
        sliceIndex++;
    }
}


bool algo::autolykos_v2::mhssamadani::isValidShare(
    algo::hash256& header,
    algo::hash256& boundary,
    uint64_t const baseNonce,
    uint32_t const baseHeight)
{
    uint8_t nonce[NONCE_SIZE_8];
    char n_str[NONCE_SIZE_8];

    uint8_t height[HEIGHT_SIZE];
    char h_str[HEIGHT_SIZE];

    *(uint64_t*)nonce = baseNonce;
    *(uint32_t*)height = baseHeight;

    ///////////////////////////////////////////////////////////////////////////
    LittleEndianToHexStr(nonce, NONCE_SIZE_8, n_str);
    for (uint32_t i = 0; i < NONCE_SIZE_8; ++i)
    {
        printf("le => nonce[%d]: %u\n", i, (uint32_t)n_str[i]);
    }
    printf("le => nonce => [%lu]\n", (*(uint64_t*)n_str));

    ///////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0; i < HEIGHT_SIZE; ++i)
    {
        printf("be => height[%d]: %u\n", i, (uint32_t)height[i]);
    }
    printf("be => height => [%u]\n", (*(uint32_t*)height));
    BigEndianToHexStr(height, HEIGHT_SIZE, h_str);
    for (uint32_t i = 0; i < HEIGHT_SIZE; ++i)
    {
        printf("le => height[%d]: %u\n", i, (uint32_t)h_str[i]);
    }
    printf("le => height => [%u]\n", (*(uint32_t*)h_str));

    ///////////////////////////////////////////////////////////////////////////
    uint8_t beN[NONCE_SIZE_8];
    HexStrToBigEndian(n_str, NONCE_SIZE_8 * 2u, beN, NONCE_SIZE_8);
    for (uint32_t i = 0u; i < NONCE_SIZE_8; ++i)
    {
        printf("hexbe => beN[%d]: %u\n", i, (uint32_t)beN[i]);
    }

    ///////////////////////////////////////////////////////////////////////////
    uint8_t beH[HEIGHT_SIZE];
    HexStrToBigEndian(h_str, HEIGHT_SIZE * 2, beH, HEIGHT_SIZE);
    for (uint32_t i = 0; i < HEIGHT_SIZE; ++i)
    {
        printf("hexbe => h_str[%d]: %u\n", i, (uint32_t)beH[i]);
    }


    ///////////////////////////////////////////////////////////////////////////
    uint8_t h1[NUM_SIZE_8];
    uint8_t m_n[NUM_SIZE_8 + NONCE_SIZE_8];
    memcpy(m_n, header.ubytes, NUM_SIZE_8);
    memcpy(m_n + NUM_SIZE_8, beN, NONCE_SIZE_8);
    for (uint32_t i = 0; i < NUM_SIZE_8; ++i)
    {
        printf("m_n[%d]: %u\n", i, (uint32_t)m_n[i]);
    }

    hashFn((const char *)m_n, NUM_SIZE_8 + NONCE_SIZE_8, (uint8_t *)h1);
    for (uint32_t i = 0; i < NUM_SIZE_8; ++i)
    {
        printf("h1[%d]: %u\n", i, (uint32_t)h1[i]);
    }

    ///////////////////////////////////////////////////////////////////////////
    uint64_t h2;
    char tmpL1[8];
    tmpL1[0] = h1[31];
    tmpL1[1] = h1[30];
    tmpL1[2] = h1[29];
    tmpL1[3] = h1[28];
    tmpL1[4] = h1[27];
    tmpL1[5] = h1[26];
    tmpL1[6] = h1[25];
    tmpL1[7] = h1[24];
    memcpy(&h2, tmpL1, 8);

    ///////////////////////////////////////////////////////////////////////////
    uint32_t HH;
    memcpy(&HH,beH,HEIGHT_SIZE);
    uint32_t N_LEN = calcN(HH);
    unsigned int h3 = h2 % N_LEN;

    ///////////////////////////////////////////////////////////////////////////
    uint8_t iii[4];
    iii[0] = ((char *)(&h3))[3];
    iii[1] = ((char *)(&h3))[2];
    iii[2] = ((char *)(&h3))[1];
    iii[3] = ((char *)(&h3))[0];

    ///////////////////////////////////////////////////////////////////////////
    uint8_t i_h_M[HEIGHT_SIZE + HEIGHT_SIZE + CONST_MES_SIZE_8];
    unsigned long long CONST_MESS[CONST_MES_SIZE_8 / 8];
    memcpy(i_h_M, iii, HEIGHT_SIZE);
    memcpy(i_h_M + HEIGHT_SIZE, beH, HEIGHT_SIZE);
    memcpy(i_h_M + HEIGHT_SIZE + HEIGHT_SIZE, CONST_MESS, CONST_MES_SIZE_8);
    hashFn((const char *)i_h_M, HEIGHT_SIZE + HEIGHT_SIZE + CONST_MES_SIZE_8, (uint8_t *)h1);
    uint8_t ff[NUM_SIZE_8 - 1];
    memcpy(ff, h1 + 1, NUM_SIZE_8 - 1);

    ///////////////////////////////////////////////////////////////////////////
    uint8_t seed[NUM_SIZE_8 - 1 + NUM_SIZE_8 + NONCE_SIZE_8];
    memcpy(seed, ff, NUM_SIZE_8 - 1);
    memcpy(seed + NUM_SIZE_8 - 1, header.ubytes, NUM_SIZE_8);
    memcpy(seed + NUM_SIZE_8 - 1 + NUM_SIZE_8, beN, NONCE_SIZE_8);

    ///////////////////////////////////////////////////////////////////////////
    uint32_t index[K_LEN];
    GenIdex((const char*)seed, NUM_SIZE_8 - 1 + NUM_SIZE_8 + NONCE_SIZE_8, index, N_LEN);
    for (uint32_t i = 0u; i < K_LEN; ++i)
    {
        printf("index[%u]: %u\n", i, index[i]);
    }


    ///////////////////////////////////////////////////////////////////////////
    uint8_t ret[32][NUM_SIZE_8];
    int ll = sizeof(uint32_t) + CONST_MES_SIZE_8 + PK_SIZE_8 + NUM_SIZE_8 + PK_SIZE_8;

    ///////////////////////////////////////////////////////////////////////////
    BIGNUM* bigsum = BN_new();
    CALL(BN_dec2bn(&bigsum, "0"), ERROR_OPENSSL);

    ///////////////////////////////////////////////////////////////////////////
    BIGNUM* bigres = BN_new();
    CALL(BN_dec2bn(&bigres, "0"), ERROR_OPENSSL);

    ///////////////////////////////////////////////////////////////////////////
    int rep = 0;
    int off = 0;
    uint8_t tmp[NUM_SIZE_8 - 1];
    uint8_t tmp2[4];
    uint8_t tmp1[4];
    unsigned char f[32];
    uint8_t Hinput[sizeof(uint32_t) + CONST_MES_SIZE_8 + PK_SIZE_8 + NUM_SIZE_8 + PK_SIZE_8];
    memset(f, 0, 32);
    for (rep = 0; rep < 32; rep++)
    {
        memset(Hinput, 0, ll);

        memcpy(tmp1, &index[rep], 4);
        tmp2[0] = tmp1[3];
        tmp2[1] = tmp1[2];
        tmp2[2] = tmp1[1];
        tmp2[3] = tmp1[0];

        off = 0;
        memcpy(Hinput + off, tmp2, sizeof(uint32_t));
        off += sizeof(uint32_t);

        memcpy(Hinput + off, beH, HEIGHT_SIZE);
        off += HEIGHT_SIZE;

        memcpy(Hinput + off, CONST_MESS, CONST_MES_SIZE_8);
        off += CONST_MES_SIZE_8;

        hashFn((const char *)Hinput, off, (uint8_t *)ret[rep]);

        memcpy(tmp, &(ret[rep][1]), 31);

        CALL(BN_bin2bn((const unsigned char *)tmp, 31, bigres), ERROR_OPENSSL);

        CALL(BN_add(bigsum, bigsum, bigres), ERROR_OPENSSL);

        BN_bn2bin(bigsum, f);
    }

    ///////////////////////////////////////////////////////////////////////////
    BN_bn2bin(bigsum, f);
    char bigendian2littl[32];
    for (size_t i = 0; i < 32; i++)
    {
        bigendian2littl[i] = f[32 - i - 1];
    }

    ///////////////////////////////////////////////////////////////////////////
    BIGNUM* littleF = BN_new();
    CALL(BN_bin2bn((const unsigned char *)bigendian2littl, 32, littleF), ERROR_OPENSSL);

    ///////////////////////////////////////////////////////////////////////////
    char hf[32];
    hashFn((const char *)f, 32, (uint8_t *)hf, true);

    ///////////////////////////////////////////////////////////////////////////
    BIGNUM* bigHF = BN_new();
    CALL(BN_bin2bn((const unsigned char *)hf, 32, bigHF), ERROR_OPENSSL);

    ///////////////////////////////////////////////////////////////////////////
    char littl2big[32];
    for (size_t i = 0; i < 32; i++)
    {
        littl2big[i] = boundary.ubytes[32 - i - 1];
    }

    ///////////////////////////////////////////////////////////////////////////
    BIGNUM* bigB = BN_new();
    CALL(BN_bin2bn((const unsigned char *)littl2big, 32, bigB), ERROR_OPENSSL);

    for (size_t i = 0; i < 32; i++)
    {
        logInfo() << "bPool[" << i << "] " << (uint32_t)boundary.ubytes[i];
    }
    for (size_t i = 0; i < 32; i++)
    {
        logInfo() << "littl2big[" << i << "] " << (uint32_t)littl2big[i];
    }
    for (size_t i = 0; i < 32; i++)
    {
        logInfo() << "hf[" << i << "] " << (uint32_t)hf[i];
    }

    int const cmp = BN_cmp(bigHF, bigB);

    BN_free(bigsum);
    BN_free(bigres);
    BN_free(littleF);
    BN_free(bigHF);
    BN_free(bigB);

    if (cmp < 0)
    {
        return true;
    }

    return false;
}
