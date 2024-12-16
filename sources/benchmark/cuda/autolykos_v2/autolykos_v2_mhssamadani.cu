///////////////////////////////////////////////////////////////////////////////
#include <algo/autolykos/autolykos.hpp>
#include <common/cast.hpp>
#include <benchmark/cuda/kernels.hpp>

///////////////////////////////////////////////////////////////////////////////
#include <benchmark/cuda/common/common.cuh>
#include <benchmark/cuda/blake2b/mhssamadani_blake2b.cuh>


__constant__ uint32_t bound_[8];


__constant__ uint64_t ivals[8] =
{
    0x6A09E667F2BDC928,
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,
    0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B,
    0x5BE0CD19137E2179
};


__device__ __forceinline__
uint64_t devROTR64(uint64_t b, int offset)
{
    uint2 a;
    uint2 result;
    a = vectorize(b);

    if (offset < 32)
    {
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
    }
    else
    {
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
        asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
    }
    return devectorize(result);
}


__device__ __forceinline__
void devB2B_G(
    uint64_t* v,
    int a,
    int b,
    int c,
    int d,
    uint64_t x,
    uint64_t y)
{
    ((uint64_t *)(v))[a] += ((uint64_t *)(v))[b] + x;
    ((uint64_t *)(v))[d] = devROTR64(((uint64_t *)(v))[d] ^ ((uint64_t *)(v))[a], 32);
    ((uint64_t *)(v))[c] += ((uint64_t *)(v))[d];
    ((uint64_t *)(v))[b] = devROTR64(((uint64_t *)(v))[b] ^ ((uint64_t *)(v))[c], 24);
    ((uint64_t *)(v))[a] += ((uint64_t *)(v))[b] + y;
    ((uint64_t *)(v))[d] = devROTR64(((uint64_t *)(v))[d] ^ ((uint64_t *)(v))[a], 16);
    ((uint64_t *)(v))[c] += ((uint64_t *)(v))[d];
    ((uint64_t *)(v))[b] = devROTR64(((uint64_t *)(v))[b] ^ ((uint64_t *)(v))[c], 63);
}


__device__ __forceinline__
void devB2B_MIX(
    uint64_t* v,
    uint64_t* m)
{
    devB2B_G(v, 0, 4, 8, 12, ((uint64_t *)(m))[0], ((uint64_t *)(m))[1]);
    devB2B_G(v, 1, 5, 9, 13, ((uint64_t *)(m))[2], ((uint64_t *)(m))[3]);
    devB2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[4], ((uint64_t *)(m))[5]);
    devB2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[6], ((uint64_t *)(m))[7]);
    devB2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[8], ((uint64_t *)(m))[9]);
    devB2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[10], ((uint64_t *)(m))[11]);
    devB2B_G(v, 2, 7,  8, 13, ((uint64_t *)(m))[12], ((uint64_t *)(m))[13]);
    devB2B_G(v, 3, 4,  9, 14, ((uint64_t *)(m))[14], ((uint64_t *)(m))[15]);

    devB2B_G(v, 0, 4,  8, 12, ((uint64_t *)(m))[14], ((uint64_t *)(m))[10]);
    devB2B_G(v, 1, 5, 9, 13, ((uint64_t *)(m))[4], ((uint64_t *)(m))[8]);
    devB2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[9], ((uint64_t *)(m))[15]);
    devB2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[13], ((uint64_t *)(m))[6]);
    devB2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[1], ((uint64_t *)(m))[12]);
    devB2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[0], ((uint64_t *)(m))[2]);
    devB2B_G(v, 2, 7, 8, 13, ((uint64_t *)(m))[11], ((uint64_t *)(m))[7]);
    devB2B_G(v, 3, 4, 9, 14, ((uint64_t *)(m))[5], ((uint64_t *)(m))[3]);

    devB2B_G(v, 0, 4, 8, 12, ((uint64_t *)(m))[11], ((uint64_t *)(m))[8]);
    devB2B_G(v, 1, 5, 9, 13, ((uint64_t *)(m))[12], ((uint64_t *)(m))[0]);
    devB2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[5], ((uint64_t *)(m))[2]);
    devB2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[15], ((uint64_t *)(m))[13]);
    devB2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[10], ((uint64_t *)(m))[14]);
    devB2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[3], ((uint64_t *)(m))[6]);
    devB2B_G(v, 2, 7, 8, 13, ((uint64_t *)(m))[7], ((uint64_t *)(m))[1]);
    devB2B_G(v, 3, 4, 9, 14, ((uint64_t *)(m))[9], ((uint64_t *)(m))[4]);

    devB2B_G(v, 0, 4, 8, 12, ((uint64_t *)(m))[7], ((uint64_t *)(m))[9]);
    devB2B_G(v, 1, 5, 9, 13, ((uint64_t *)(m))[3], ((uint64_t *)(m))[1]);
    devB2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[13], ((uint64_t *)(m))[12]);
    devB2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[11], ((uint64_t *)(m))[14]);
    devB2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[2], ((uint64_t *)(m))[6]);
    devB2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[5], ((uint64_t *)(m))[10]);
    devB2B_G(v, 2, 7, 8, 13, ((uint64_t *)(m))[4], ((uint64_t *)(m))[0]);
    devB2B_G(v, 3, 4, 9, 14, ((uint64_t *)(m))[15], ((uint64_t *)(m))[8]);

    devB2B_G(v, 0, 4, 8, 12, ((uint64_t *)(m))[9], ((uint64_t *)(m))[0]);
    devB2B_G(v, 1, 5, 9, 13, ((uint64_t *)(m))[5], ((uint64_t *)(m))[7]);
    devB2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[2], ((uint64_t *)(m))[4]);
    devB2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[10], ((uint64_t *)(m))[15]);
    devB2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[14], ((uint64_t *)(m))[1]);
    devB2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[11], ((uint64_t *)(m))[12]);
    devB2B_G(v, 2, 7, 8, 13, ((uint64_t *)(m))[6], ((uint64_t *)(m))[8]);
    devB2B_G(v, 3, 4, 9, 14, ((uint64_t *)(m))[3], ((uint64_t *)(m))[13]);

    devB2B_G(v, 0, 4, 8, 12, ((uint64_t *)(m))[2], ((uint64_t *)(m))[12]);
    devB2B_G(v, 1, 5, 9, 13, ((uint64_t *)(m))[6], ((uint64_t *)(m))[10]);
    devB2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[0], ((uint64_t *)(m))[11]);
    devB2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[8], ((uint64_t *)(m))[3]);
    devB2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[4], ((uint64_t *)(m))[13]);
    devB2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[7], ((uint64_t *)(m))[5]);
    devB2B_G(v, 2, 7,  8, 13, ((uint64_t *)(m))[15], ((uint64_t *)(m))[14]);
    devB2B_G(v, 3, 4, 9, 14, ((uint64_t *)(m))[1], ((uint64_t *)(m))[9]);

    devB2B_G(v, 0, 4, 8, 12, ((uint64_t *)(m))[12], ((uint64_t *)(m))[5]);
    devB2B_G(v, 1, 5, 9, 13, ((uint64_t *)(m))[1], ((uint64_t *)(m))[15]);
    devB2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[14], ((uint64_t *)(m))[13]);
    devB2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[4], ((uint64_t *)(m))[10]);
    devB2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[0], ((uint64_t *)(m))[7]);
    devB2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[6], ((uint64_t *)(m))[3]);
    devB2B_G(v, 2, 7, 8, 13, ((uint64_t *)(m))[9], ((uint64_t *)(m))[2]);
    devB2B_G(v, 3, 4, 9, 14, ((uint64_t *)(m))[8], ((uint64_t *)(m))[11]);

    devB2B_G(v, 0, 4,  8, 12, ((uint64_t *)(m))[13], ((uint64_t *)(m))[11]);
    devB2B_G(v, 1, 5, 9, 13, ((uint64_t *)(m))[7], ((uint64_t *)(m))[14]);
    devB2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[12], ((uint64_t *)(m))[1]);
    devB2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[3], ((uint64_t *)(m))[9]);
    devB2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[5], ((uint64_t *)(m))[0]);
    devB2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[15], ((uint64_t *)(m))[4]);
    devB2B_G(v, 2, 7, 8, 13, ((uint64_t *)(m))[8], ((uint64_t *)(m))[6]);
    devB2B_G(v, 3, 4, 9, 14, ((uint64_t *)(m))[2], ((uint64_t *)(m))[10]);

    devB2B_G(v, 0, 4, 8, 12, ((uint64_t *)(m))[6], ((uint64_t *)(m))[15]);
    devB2B_G(v, 1, 5, 9, 13, ((uint64_t *)(m))[14], ((uint64_t *)(m))[9]);
    devB2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[11], ((uint64_t *)(m))[3]);
    devB2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[0], ((uint64_t *)(m))[8]);
    devB2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[12], ((uint64_t *)(m))[2]);
    devB2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[13], ((uint64_t *)(m))[7]);
    devB2B_G(v, 2, 7, 8, 13, ((uint64_t *)(m))[1], ((uint64_t *)(m))[4]);
    devB2B_G(v, 3, 4, 9, 14, ((uint64_t *)(m))[10], ((uint64_t *)(m))[5]);

    devB2B_G(v, 0, 4, 8, 12, ((uint64_t *)(m))[10], ((uint64_t *)(m))[2]);
    devB2B_G(v, 1, 5, 9, 13, ((uint64_t *)(m))[8], ((uint64_t *)(m))[4]);
    devB2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[7], ((uint64_t *)(m))[6]);
    devB2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[1], ((uint64_t *)(m))[5]);
    devB2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[15], ((uint64_t *)(m))[11]);
    devB2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[9], ((uint64_t *)(m))[14]);
    devB2B_G(v, 2, 7, 8, 13, ((uint64_t *)(m))[3], ((uint64_t *)(m))[12]);
    devB2B_G(v, 3, 4, 9, 14, ((uint64_t *)(m))[13], ((uint64_t *)(m))[0]);

    devB2B_G(v, 0, 4, 8, 12, ((uint64_t *)(m))[0], ((uint64_t *)(m))[1]);
    devB2B_G(v, 1, 5, 9, 13, ((uint64_t *)(m))[2], ((uint64_t *)(m))[3]);
    devB2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[4], ((uint64_t *)(m))[5]);
    devB2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[6], ((uint64_t *)(m))[7]);
    devB2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[8], ((uint64_t *)(m))[9]);
    devB2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[10], ((uint64_t *)(m))[11]);
    devB2B_G(v, 2, 7,  8, 13, ((uint64_t *)(m))[12], ((uint64_t *)(m))[13]);
    devB2B_G(v, 3, 4,  9, 14, ((uint64_t *)(m))[14], ((uint64_t *)(m))[15]);

    devB2B_G(v, 0, 4,  8, 12, ((uint64_t *)(m))[14], ((uint64_t *)(m))[10]);
    devB2B_G(v, 1, 5, 9, 13, ((uint64_t *)(m))[4], ((uint64_t *)(m))[8]);
    devB2B_G(v, 2, 6, 10, 14, ((uint64_t *)(m))[9], ((uint64_t *)(m))[15]);
    devB2B_G(v, 3, 7, 11, 15, ((uint64_t *)(m))[13], ((uint64_t *)(m))[6]);
    devB2B_G(v, 0, 5, 10, 15, ((uint64_t *)(m))[1], ((uint64_t *)(m))[12]);
    devB2B_G(v, 1, 6, 11, 12, ((uint64_t *)(m))[0], ((uint64_t *)(m))[2]);
    devB2B_G(v, 2, 7, 8, 13, ((uint64_t *)(m))[11], ((uint64_t *)(m))[7]);
    devB2B_G(v, 3, 4, 9, 14, ((uint64_t *)(m))[5], ((uint64_t *)(m))[3]);
}


__global__ __launch_bounds__(64, 64)
__global__ void BlockMiningStep1(
    uint32_t const* data,   // Header
    uint32_t const* hashes, // precalculated hashes
    uint32_t* BHashes,      // Will be fill
    uint32_t const n_len,   // period
    uint64_t const base     // nonce base
)
{

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t r[9] = { 0 };

    uint64_t aux[32];
    uint32_t non[algo::autolykos_v2::NONCE_SIZE_32];

    uint32_t j;
    uint64_t tmp;
    uint64_t hsh;
    uint64_t h2;
    uint32_t h3;

    for (int ii = 0; ii < 4; ii++)
    {
        tid = (algo::autolykos_v2::NONCES_PER_ITER / 4u) * ii + threadIdx.x + blockDim.x * blockIdx.x;
        if (tid < algo::autolykos_v2::NONCES_PER_ITER)
        {
            asm volatile
            (
                "add.cc.u32 %0, %1, %2;":
                "=r"(non[0]) : "r"(((uint32_t *)&base)[0]), "r"(tid)
            );

            asm volatile("addc.u32 %0, %1, 0;": "=r"(non[1]) : "r"(((uint32_t *)&base)[1]));

            ((uint32_t*)(&tmp))[0] = __byte_perm(non[1], 0, 0x0123);
            ((uint32_t*)(&tmp))[1] = __byte_perm(non[0], 0, 0x0123);

            B2B_IV(aux);
            B2B_IV(aux + 8);
            aux[0] = ivals[0];
            ((uint64_t*)(aux))[12] ^= 40;
            ((uint64_t*)(aux))[13] ^= 0;

            ((uint64_t*)(aux))[14] = ~((uint64_t*)(aux))[14];
            ((uint64_t*)(aux))[16] =  ((uint64_t*)data)[0];
            ((uint64_t*)(aux))[17] =  ((uint64_t*)data)[1];
            ((uint64_t*)(aux))[18] =  ((uint64_t*)data)[2];
            ((uint64_t*)(aux))[19] =  ((uint64_t*)data)[3];
            ((uint64_t*)(aux))[20] =  tmp;
            ((uint64_t*)(aux))[21] =  0;
            ((uint64_t*)(aux))[22] =  0;
            ((uint64_t*)(aux))[23] =  0;
            ((uint64_t*)(aux))[24] =  0;
            ((uint64_t*)(aux))[25] =  0;
            ((uint64_t*)(aux))[26] =  0;
            ((uint64_t*)(aux))[27] =  0;
            ((uint64_t*)(aux))[28] =  0;
            ((uint64_t*)(aux))[29] =  0;
            ((uint64_t*)(aux))[30] =  0;
            ((uint64_t*)(aux))[31] =  0;

            devB2B_MIX(aux, aux + 16);


            #pragma unroll
            for (j = 0; j < algo::autolykos_v2::NUM_SIZE_32; j += 2)
            {
                hsh = ivals[j >> 1];
                hsh ^= ((uint64_t *)(aux))[j >> 1] ^ ((uint64_t *)(aux))[8 + (j >> 1)];

                r[j] =  ((uint32_t*)(&hsh))[0];
                r[j + 1] = ((uint32_t*)(&hsh))[1];
            }

            //----------------------------------------------------------------------------------------
            ((uint8_t*)&h2)[0] = ((uint8_t*)r)[31];
            ((uint8_t*)&h2)[1] = ((uint8_t*)r)[30];
            ((uint8_t*)&h2)[2] = ((uint8_t*)r)[29];
            ((uint8_t*)&h2)[3] = ((uint8_t*)r)[28];
            ((uint8_t*)&h2)[4] = ((uint8_t*)r)[27];
            ((uint8_t*)&h2)[5] = ((uint8_t*)r)[26];
            ((uint8_t*)&h2)[6] = ((uint8_t*)r)[25];
            ((uint8_t*)&h2)[7] = ((uint8_t*)r)[24];

            h3 = h2 % n_len;

            #pragma unroll 8
            for (int i = 0; i < 8; ++i)
            {
                    r[7 - i] = be_u32(hashes[(h3 << 3) + i]);
            }

            //------------------------------------------------------
            B2B_IV(aux);
            B2B_IV(aux + 8);

            aux[0] = ivals[0];

            ((uint64_t *)(aux))[12] ^= 71; //31+32+8;
            ((uint64_t *)(aux))[13] ^= 0;

            ((uint64_t *)(aux))[14] = ~((uint64_t *)(aux))[14];

            uint8_t *bb = (uint8_t *)(&(((uint64_t *)(aux))[16]));
            ((uint64_t *)bb)[0] = ((uint64_t *)(&((uint8_t *)r)[1]))[0];
            ((uint64_t *)bb)[1] = ((uint64_t *)(&((uint8_t *)r)[1]))[1];
            ((uint64_t *)bb)[2] = ((uint64_t *)(&((uint8_t *)r)[1]))[2];
            ((uint64_t *)bb)[3] = ((uint64_t *)(&((uint8_t *)r)[1]))[3];

            ((uint64_t *)&bb[31])[0] = ((uint64_t *)data)[0];
            ((uint64_t *)&bb[39])[0] = ((uint64_t *)data)[1];
            ((uint64_t *)&bb[47])[0] = ((uint64_t *)data)[2];
            ((uint64_t *)&bb[55])[0] = ((uint64_t *)data)[3];

            ((uint64_t *)&bb[63])[0] = tmp;

            ((uint64_t *)(aux))[25] = 0;
            ((uint64_t *)(aux))[26] = 0;
            ((uint64_t *)(aux))[27] = 0;
            ((uint64_t *)(aux))[28] = 0;
            ((uint64_t *)(aux))[29] = 0;
            ((uint64_t *)(aux))[30] = 0;
            ((uint64_t *)(aux))[31] = 0;

            devB2B_MIX(aux, aux + 16);

            #pragma unroll
            for (j = 0; j < algo::autolykos_v2::NUM_SIZE_32; j += 2)
            {
                hsh = ivals[j >> 1];
                hsh ^= ((uint64_t *)(aux))[j >> 1] ^ ((uint64_t *)(aux))[8 + (j >> 1)];
                BHashes[algo::autolykos_v2::THREADS_PER_ITER * j + tid] = __byte_perm(((uint32_t*)(&hsh))[0], 0, 0x0123);
                BHashes[algo::autolykos_v2::THREADS_PER_ITER * (j + 1) + tid] = __byte_perm(((uint32_t*)(&hsh))[1], 0, 0x0123);
            }
        }
    }
}


__global__ __launch_bounds__(64, 64)
__global__ void BlockMiningStep2(
    t_result_64* result,    // result
    uint32_t const* data,   // Header
    uint32_t const* hashes, // precalculated hashes
    uint32_t*  BHashes,     // From BlockMiningStep1
    uint32_t const n_len,   // period
    uint32_t const height,  // block height
    uint64_t const base     // nonce
)
{
    __shared__ uint32_t shared_index[64];
    __shared__ uint32_t shared_data[512];

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t const thread_id = threadIdx.x & 7;
    uint32_t const thrdblck_id = threadIdx.x;
    uint32_t const hash_id = threadIdx.x >> 3;

    uint64_t aux[32] = { 0 };
    uint32_t ind[32] = { 0 };
    uint32_t r[9] = { 0 };

    uint4 v1 = { 0, 0, 0, 0 };
    uint4 v2 = { 0, 0, 0, 0 };
    uint4 v3 = { 0, 0, 0, 0 };
    uint4 v4 = { 0, 0, 0, 0 };

    uint8_t j = 0;

    if (tid < algo::autolykos_v2::NONCES_PER_ITER)
    {
        #pragma unroll
        for (int k = 0; k < 8; k++)
        {
            r[k] = (BHashes[k *  algo::autolykos_v2::THREADS_PER_ITER + tid]);
        }

        //================================================================//
        //  Generate indices
        //================================================================//
        ((uint8_t *)r)[32] = ((uint8_t *)r)[0];
        ((uint8_t *)r)[33] = ((uint8_t *)r)[1];
        ((uint8_t *)r)[34] = ((uint8_t *)r)[2];
        ((uint8_t *)r)[35] = ((uint8_t *)r)[3];

        #pragma unroll
        for (int k = 0; k < algo::autolykos_v2::K_LEN; k += 4)
        {
            ind[k] = r[k >> 2] % n_len;
            ind[k + 1] = ((r[k >> 2] << 8) | (r[(k >> 2) + 1] >> 24)) % n_len;
            ind[k + 2] = ((r[k >> 2] << 16) | (r[(k >> 2) + 1] >> 16)) % n_len;
            ind[k + 3] = ((r[k >> 2] << 24) | (r[(k >> 2) + 1] >> 8)) % n_len;
        }

        //================================================================//
        //  Calculate result
        //================================================================//
        shared_index[thrdblck_id] = ind[0];
        __syncthreads();

        shared_data[(hash_id << 3) + thread_id] = (hashes[(shared_index[hash_id] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 64] = (hashes[(shared_index[hash_id + 8] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 128] = (hashes[(shared_index[hash_id + 16] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 192] = (hashes[(shared_index[hash_id + 24] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 256] = (hashes[(shared_index[hash_id + 32] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 320] = (hashes[(shared_index[hash_id + 40] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 384] = (hashes[(shared_index[hash_id + 48] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 448] = (hashes[(shared_index[hash_id + 56] << 3) + thread_id]);
        __syncthreads();

        v1.x = shared_data[(thrdblck_id << 3) + 0];
        v1.y = shared_data[(thrdblck_id << 3) + 1];
        v1.z = shared_data[(thrdblck_id << 3) + 2];
        v1.w = shared_data[(thrdblck_id << 3) + 3];
        v3.x = shared_data[(thrdblck_id << 3) + 4];
        v3.y = shared_data[(thrdblck_id << 3) + 5];
        v3.z = shared_data[(thrdblck_id << 3) + 6];
        v3.w = shared_data[(thrdblck_id << 3) + 7];

        shared_index[thrdblck_id] = ind[1];
        __syncthreads();

        shared_data[(hash_id << 3) + thread_id] = (hashes[(shared_index[hash_id] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 64] = (hashes[(shared_index[hash_id + 8] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 128] = (hashes[(shared_index[hash_id + 16] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 192] = (hashes[(shared_index[hash_id + 24] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 256] = (hashes[(shared_index[hash_id + 32] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 320] = (hashes[(shared_index[hash_id + 40] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 384] = (hashes[(shared_index[hash_id + 48] << 3) + thread_id]);
        shared_data[(hash_id << 3) + thread_id + 448] = (hashes[(shared_index[hash_id + 56] << 3) + thread_id]);
        __syncthreads();

        v2.x = shared_data[(thrdblck_id << 3) + 0];
        v2.y = shared_data[(thrdblck_id << 3) + 1];
        v2.z = shared_data[(thrdblck_id << 3) + 2];
        v2.w = shared_data[(thrdblck_id << 3) + 3];
        v4.x = shared_data[(thrdblck_id << 3) + 4];
        v4.y = shared_data[(thrdblck_id << 3) + 5];
        v4.z = shared_data[(thrdblck_id << 3) + 6];
        v4.w = shared_data[(thrdblck_id << 3) + 7];

        asm volatile ("add.cc.u32 %0, %1, %2;":"=r"(r[0]) : "r"(v1.x), "r"(v2.x));
        asm volatile ("addc.cc.u32 %0, %1, %2;":"=r"(r[1]) : "r"(v1.y), "r"(v2.y));
        asm volatile ("addc.cc.u32 %0, %1, %2;":"=r"(r[2]) : "r"(v1.z), "r"(v2.z));
        asm volatile ("addc.cc.u32 %0, %1, %2;":"=r"(r[3]) : "r"(v1.w), "r"(v2.w));
        asm volatile ("addc.cc.u32 %0, %1, %2;":"=r"(r[4]) : "r"(v3.x), "r"(v4.x));
        asm volatile ("addc.cc.u32 %0, %1, %2;":"=r"(r[5]) : "r"(v3.y), "r"(v4.y));
        asm volatile ("addc.cc.u32 %0, %1, %2;":"=r"(r[6]) : "r"(v3.z), "r"(v4.z));
        asm volatile ("addc.cc.u32 %0, %1, %2;":"=r"(r[7]) : "r"(v3.w), "r"(v4.w));
        asm volatile ("addc.u32 %0, 0, 0;": "=r"(r[8]));

        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        // remaining additions
        #pragma unroll
        for (int k = 2; k < algo::autolykos_v2::K_LEN; ++k)
        {
            shared_index[thrdblck_id] = ind[k];
            __syncthreads();

            shared_data[(hash_id << 3) + thread_id] = (hashes[(shared_index[hash_id] << 3) + thread_id]);
            shared_data[(hash_id << 3) + thread_id + 64] = (hashes[(shared_index[hash_id + 8] << 3) + thread_id]);
            shared_data[(hash_id << 3) + thread_id + 128] = (hashes[(shared_index[hash_id + 16] << 3) + thread_id]);
            shared_data[(hash_id << 3) + thread_id + 192] = (hashes[(shared_index[hash_id + 24] << 3) + thread_id]);
            shared_data[(hash_id << 3) + thread_id + 256] = (hashes[(shared_index[hash_id + 32] << 3) + thread_id]);
            shared_data[(hash_id << 3) + thread_id + 320] = (hashes[(shared_index[hash_id + 40] << 3) + thread_id]);
            shared_data[(hash_id << 3) + thread_id + 384] = (hashes[(shared_index[hash_id + 48] << 3) + thread_id]);
            shared_data[(hash_id << 3) + thread_id + 448] = (hashes[(shared_index[hash_id + 56] << 3) + thread_id]);
            __syncthreads();

            v1.x = shared_data[(thrdblck_id << 3) + 0];
            v1.y = shared_data[(thrdblck_id << 3) + 1];
            v1.z = shared_data[(thrdblck_id << 3) + 2];
            v1.w = shared_data[(thrdblck_id << 3) + 3];
            v2.x = shared_data[(thrdblck_id << 3) + 4];
            v2.y = shared_data[(thrdblck_id << 3) + 5];
            v2.z = shared_data[(thrdblck_id << 3) + 6];
            v2.w = shared_data[(thrdblck_id << 3) + 7];

            asm volatile ("add.cc.u32 %0, %0, %1;":"+r"(r[0]) : "r"(v1.x));
            asm volatile ("addc.cc.u32 %0, %0, %1;":"+r"(r[1]) : "r"(v1.y));
            asm volatile ("addc.cc.u32 %0, %0, %1;":"+r"(r[2]) : "r"(v1.z));
            asm volatile ("addc.cc.u32 %0, %0, %1;":"+r"(r[3]) : "r"(v1.w));
            asm volatile ("addc.cc.u32 %0, %0, %1;":"+r"(r[4]) : "r"(v2.x));
            asm volatile ("addc.cc.u32 %0, %0, %1;":"+r"(r[5]) : "r"(v2.y));
            asm volatile ("addc.cc.u32 %0, %0, %1;":"+r"(r[6]) : "r"(v2.z));
            asm volatile ("addc.cc.u32 %0, %0, %1;":"+r"(r[7]) : "r"(v2.w));
            asm volatile ("addc.u32 %0, %0, 0;": "+r"(r[8]));
        }

        B2B_IV(aux);
        B2B_IV(aux + 8);
        aux[0] = ivals[0];
        ((uint64_t*)(aux))[12] ^= 32;
        ((uint64_t*)(aux))[13] ^= 0;

        ((uint64_t*)(aux))[14] = ~((uint64_t*)(aux))[14];

        uint8_t* bb = (uint8_t*)(&(((uint64_t*)(aux))[16]));
        for (j = 0; j < algo::autolykos_v2::NUM_SIZE_8; ++j)
        {
            bb[j] = ((const uint8_t *)r)[algo::autolykos_v2::NUM_SIZE_8 - j - 1];
        }

        ((uint64_t *)(aux))[20] = 0;
        ((uint64_t *)(aux))[21] = 0;
        ((uint64_t *)(aux))[22] = 0;
        ((uint64_t *)(aux))[23] = 0;
        ((uint64_t *)(aux))[24] = 0;
        ((uint64_t *)(aux))[25] = 0;
        ((uint64_t *)(aux))[26] = 0;
        ((uint64_t *)(aux))[27] = 0;
        ((uint64_t *)(aux))[28] = 0;
        ((uint64_t *)(aux))[29] = 0;
        ((uint64_t *)(aux))[30] = 0;
        ((uint64_t *)(aux))[31] = 0;

        devB2B_MIX(aux, aux + 16);

        uint64_t hsh;
        uint32_t r_l[32];
        #pragma unroll
        for (j = 0; j < algo::autolykos_v2::NUM_SIZE_32; j += 2)
        {
            hsh = ivals[j >> 1];
            hsh ^= ((uint64_t *)(aux))[j >> 1] ^ ((uint64_t *)(aux))[8 + (j >> 1)];
            r_l[j] = ((uint32_t*)&hsh)[0];
            r_l[j+1] = ((uint32_t*)&hsh)[1];

        }

        #pragma unroll 32
        for (j = 0; j < algo::autolykos_v2::NUM_SIZE_8; j ++)
        {
            ((uint8_t *)r)[j] = ((uint8_t *)r_l)[algo::autolykos_v2::NUM_SIZE_8 - j - 1];
        }


        //================================================================//
        //  Dump result to global memory -- LITTLE ENDIAN
        //================================================================//
        j =    ((uint64_t *)r)[3] < ((uint64_t *)bound_)[3]
            || ((uint64_t *)r)[3] == ((uint64_t *)bound_)[3]
            &&
            (
                   ((uint64_t *)r)[2] < ((uint64_t *)bound_)[2]
                || ((uint64_t *)r)[2] == ((uint64_t *)bound_)[2]
                &&
                (
                       ((uint64_t *)r)[1] < ((uint64_t *)bound_)[1]
                    || ((uint64_t *)r)[1] == ((uint64_t *)bound_)[1]
                    && ((uint64_t *)r)[0] < ((uint64_t *)bound_)[0]
                )
            );

        if (j)
        {
            uint32_t const index = atomicAdd((uint32_t*)&result->index, 1);
            if (index < MAX_RESULT_INDEX)
            {
                result->found = true;
                result->nonce[index] = tid + base;
            }
        }
    }
}


__host__
bool autolykos_v2_mhssamadi_init(algo::hash256 const& boundary)
{
    CUDA_ER(cudaMemcpyToSymbol(bound_, (void*)&boundary, algo::LEN_HASH_256));

    return true;
}


__host__
bool autolykos_v2_mhssamadi(
        cudaStream_t stream,
        t_result_64* result,
        uint32_t const* dag,
        uint32_t* BHashes,
        uint32_t* header,
        uint32_t const blocks,
        uint32_t const threads,
        uint32_t const period,
        uint32_t const height)
{
    uint64_t const nonce{ 11055774138563218679ull };

    BlockMiningStep1<<<blocks / 4u, threads, 0, stream>>>
    (
        header,
        dag,
        BHashes,
        period,
        nonce
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    BlockMiningStep2<<<blocks, threads, 0, stream>>>
    (
        result,
        header,
        dag,
        BHashes,
        period,
        height,
        nonce
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}