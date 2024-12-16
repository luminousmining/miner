///////////////////////////////////////////////////////////////////////////////
#include <algo/autolykos/autolykos.hpp>
#include <common/cast.hpp>
#include <benchmark/cuda/kernels.hpp>

///////////////////////////////////////////////////////////////////////////////
#include <benchmark/cuda/common/common.cuh>


__constant__ uint32_t d_bound[8];


#define B2B_INIT(m)                                                            \
    m[0] = 0x6A09E667F3BCC908;                                                 \
    m[1] = 0xBB67AE8584CAA73B;                                                 \
    m[2] = 0x3C6EF372FE94F82B;                                                 \
    m[3] = 0xA54FF53A5F1D36F1;                                                 \
    m[4] = 0x510E527FADE682D1;                                                 \
    m[5] = 0x9B05688C2B3E6C1F;                                                 \
    m[6] = 0x1F83D9ABFB41BD6B;                                                 \
    m[7] = 0x5BE0CD19137E2179;                                                 \
                                                                               \
    m[8]  = 0x6A09E667F3BCC908;                                                \
    m[9]  = 0xBB67AE8584CAA73B;                                                \
    m[10] = 0x3C6EF372FE94F82B;                                                \
    m[11] = 0xA54FF53A5F1D36F1;                                                \
    m[12] = 0x510E527FADE682D1;                                                \
    m[13] = 0x9B05688C2B3E6C1F;                                                \
    m[14] = 0x1F83D9ABFB41BD6B;                                                \
    m[15] = 0x5BE0CD19137E2179;


#define B2B_G(a, b, c, d, m1, m2)                                              \
    v[a] += v[b] + m1;                                                         \
    v[d] = ror_64(v[d] ^ v[a], 32);                                            \
    v[c] += v[d];                                                              \
    v[b] = ror_64(v[b] ^ v[c], 24);                                            \
    v[a] += v[b] + m2;                                                         \
    v[d] = ror_64(v[d] ^ v[a], 16);                                            \
    v[c] += v[d];                                                              \
    v[b] = ror_64(v[b] ^ v[c], 63);


__device__ __forceinline__
void devB2B_MIX(
    uint64_t* const v,
    uint64_t* const m)
{
    B2B_G(0, 4, 8,  12, m[0],  m[1]);
    B2B_G(1, 5, 9,  13, m[2],  m[3]);
    B2B_G(2, 6, 10, 14, m[4],  m[5]);
    B2B_G(3, 7, 11, 15, m[6],  m[7]);
    B2B_G(0, 5, 10, 15, m[8],  m[9]);
    B2B_G(1, 6, 11, 12, m[10], m[11]);
    B2B_G(2, 7, 8,  13, m[12], m[13]);
    B2B_G(3, 4, 9,  14, m[14], m[15]);

    B2B_G(0, 4, 8,  12, m[14], m[10]);
    B2B_G(1, 5, 9,  13, m[4],  m[8]);
    B2B_G(2, 6, 10, 14, m[9],  m[15]);
    B2B_G(3, 7, 11, 15, m[13], m[6]);
    B2B_G(0, 5, 10, 15, m[1],  m[12]);
    B2B_G(1, 6, 11, 12, m[0],  m[2]);
    B2B_G(2, 7, 8,  13, m[11], m[7]);
    B2B_G(3, 4, 9,  14, m[5],  m[3]);

    B2B_G(0, 4, 8,  12, m[11], m[8]);
    B2B_G(1, 5, 9,  13, m[12], m[0]);
    B2B_G(2, 6, 10, 14, m[5],  m[2]);
    B2B_G(3, 7, 11, 15, m[15], m[13]);
    B2B_G(0, 5, 10, 15, m[10], m[14]);
    B2B_G(1, 6, 11, 12, m[3],  m[6]);
    B2B_G(2, 7, 8,  13, m[7],  m[1]);
    B2B_G(3, 4, 9,  14, m[9],  m[4]);

    B2B_G(0, 4, 8,  12, m[7],  m[9]);
    B2B_G(1, 5, 9,  13, m[3],  m[1]);
    B2B_G(2, 6, 10, 14, m[13], m[12]);
    B2B_G(3, 7, 11, 15, m[11], m[14]);
    B2B_G(0, 5, 10, 15, m[2],  m[6]);
    B2B_G(1, 6, 11, 12, m[5],  m[10]);
    B2B_G(2, 7, 8,  13, m[4],  m[0]);
    B2B_G(3, 4, 9,  14, m[15], m[8]);

    B2B_G(0, 4, 8,  12, m[9],  m[0]);
    B2B_G(1, 5, 9,  13, m[5],  m[7]);
    B2B_G(2, 6, 10, 14, m[2],  m[4]);
    B2B_G(3, 7, 11, 15, m[10], m[15]);
    B2B_G(0, 5, 10, 15, m[14], m[1]);
    B2B_G(1, 6, 11, 12, m[11], m[12]);
    B2B_G(2, 7, 8,  13, m[6],  m[8]);
    B2B_G(3, 4, 9,  14, m[3],  m[13]);

    B2B_G(0, 4, 8,  12, m[2],  m[12]);
    B2B_G(1, 5, 9,  13, m[6],  m[10]);
    B2B_G(2, 6, 10, 14, m[0],  m[11]);
    B2B_G(3, 7, 11, 15, m[8],  m[3]);
    B2B_G(0, 5, 10, 15, m[4],  m[13]);
    B2B_G(1, 6, 11, 12, m[7],  m[5]);
    B2B_G(2, 7, 8,  13, m[15], m[14]);
    B2B_G(3, 4, 9,  14, m[1],  m[9]);

    B2B_G(0, 4, 8,  12, m[12], m[5]);
    B2B_G(1, 5, 9,  13, m[1],  m[15]);
    B2B_G(2, 6, 10, 14, m[14], m[13]);
    B2B_G(3, 7, 11, 15, m[4],  m[10]);
    B2B_G(0, 5, 10, 15, m[0],  m[7]);
    B2B_G(1, 6, 11, 12, m[6],  m[3]);
    B2B_G(2, 7, 8,  13, m[9],  m[2]);
    B2B_G(3, 4, 9,  14, m[8],  m[11]);

    B2B_G(0, 4, 8,  12, m[13], m[11]);
    B2B_G(1, 5, 9,  13, m[7],  m[14]);
    B2B_G(2, 6, 10, 14, m[12], m[1]);
    B2B_G(3, 7, 11, 15, m[3],  m[9]);
    B2B_G(0, 5, 10, 15, m[5],  m[0]);
    B2B_G(1, 6, 11, 12, m[15], m[4]);
    B2B_G(2, 7, 8,  13, m[8],  m[6]);
    B2B_G(3, 4, 9,  14, m[2],  m[10]);

    B2B_G(0, 4, 8,  12, m[6],  m[15]);
    B2B_G(1, 5, 9,  13, m[14], m[9]);
    B2B_G(2, 6, 10, 14, m[11], m[3]);
    B2B_G(3, 7, 11, 15, m[0],  m[8]);
    B2B_G(0, 5, 10, 15, m[12], m[2]);
    B2B_G(1, 6, 11, 12, m[13], m[7]);
    B2B_G(2, 7, 8,  13, m[1],  m[4]);
    B2B_G(3, 4, 9,  14, m[10], m[5]);

    B2B_G(0, 4, 8,  12, m[10], m[2]);
    B2B_G(1, 5, 9,  13, m[8],  m[4]);
    B2B_G(2, 6, 10, 14, m[7],  m[6]);
    B2B_G(3, 7, 11, 15, m[1],  m[5]);
    B2B_G(0, 5, 10, 15, m[15], m[11]);
    B2B_G(1, 6, 11, 12, m[9],  m[14]);
    B2B_G(2, 7, 8,  13, m[3],  m[12]);
    B2B_G(3, 4, 9,  14, m[13], m[0]);

    B2B_G(0, 4, 8,  12, m[0],  m[1]);
    B2B_G(1, 5, 9,  13, m[2],  m[3]);
    B2B_G(2, 6, 10, 14, m[4],  m[5]);
    B2B_G(3, 7, 11, 15, m[6],  m[7]);
    B2B_G(0, 5, 10, 15, m[8],  m[9]);
    B2B_G(1, 6, 11, 12, m[10], m[11]);
    B2B_G(2, 7, 8,  13, m[12], m[13]);
    B2B_G(3, 4, 9,  14, m[14], m[15]);

    B2B_G(0, 4, 8,  12, m[14], m[10]);
    B2B_G(1, 5, 9,  13, m[4],  m[8]);
    B2B_G(2, 6, 10, 14, m[9],  m[15]);
    B2B_G(3, 7, 11, 15, m[13], m[6]);
    B2B_G(0, 5, 10, 15, m[1],  m[12]);
    B2B_G(1, 6, 11, 12, m[0],  m[2]);
    B2B_G(2, 7, 8,  13, m[11], m[7]);
    B2B_G(3, 4, 9,  14, m[5],  m[3]);
}


__global__
void __launch_bounds__(64, 64)
kernel_autolykos_v2_step1_v1(
    uint32_t* __restrict__              hashes,
    uint64_t const* __restrict__ const  header,
    uint32_t* __restrict__ const        BHashes,
    uint64_t const                      nonce,
    uint32_t const                      period)
{
    uint32_t r[9];
    uint64_t aux[32];
    uint32_t non[algo::autolykos_v2::NONCE_SIZE_32];

    uint32_t tid;
    uint64_t tmp;
    uint64_t hsh;
    uint64_t h2;
    uint32_t h3;

    ///////////////////////////////////////////////////////////////////////////
    #pragma unroll 1
    for (uint32_t ii = 0u; ii < 4; ++ii)
    {
        tid = ((algo::autolykos_v2::NONCES_PER_ITER / 4u) * ii)
            + (threadIdx.x + (blockDim.x * blockIdx.x));

        if (tid >= algo::autolykos_v2::NONCES_PER_ITER)
        {
            break;
        }

        asm volatile(
            "add.cc.u32 %0, %1, %2;"
            :"=r"(non[0])
            :"r"(((uint32_t *)&nonce)[0]),
            "r"(tid));
        asm volatile(
            "addc.u32 %0, %1, 0;"
            :"=r"(non[1])
            :"r"(((uint32_t *)&nonce)[1]));

        ((uint32_t*)(&tmp))[0] = be_u32(non[1], 0);
        ((uint32_t*)(&tmp))[1] = be_u32(non[0], 0);

        B2B_INIT(aux);
        aux[0] = 0x6A09E667F2BDC928;
        ((uint64_t *)(aux))[12] ^= 40;
        ((uint64_t *)(aux))[13] ^= 0;

        aux[14] = ~(aux[14]);
        aux[16] = header[0];
        aux[17] = header[1];
        aux[18] = header[2];
        aux[19] = header[3];

        aux[20] = tmp;
        aux[21] = 0ull;
        aux[22] = 0ull;
        aux[23] = 0ull;
        aux[24] = 0ull;
        aux[25] = 0ull;
        aux[26] = 0ull;
        aux[27] = 0ull;
        aux[28] = 0ull;
        aux[29] = 0ull;
        aux[30] = 0ull;
        aux[31] = 0ull;

        ///////////////////////////////////////////////////////////////////////////
        devB2B_MIX(aux, aux + 16);

        ///////////////////////////////////////////////////////////////////////////
        {
            hsh = 0x6A09E667F2BDC928;
            hsh ^= aux[0] ^ aux[8];
            r[0] = (uint32_t)hsh;
            r[1] = (uint32_t)(hsh >> 32);
        }
        {
            hsh = 0xBB67AE8584CAA73B;
            hsh ^= aux[1] ^ aux[9];
            r[2] = (uint32_t)hsh;
            r[3] = (uint32_t)(hsh >> 32);
        }
        {
            hsh = 0x3C6EF372FE94F82B;
            hsh ^= aux[2] ^ aux[10];
            r[4] = (uint32_t)hsh;
            r[5] = (uint32_t)(hsh >> 32);
        }
        {
            hsh = 0xA54FF53A5F1D36F1;
            hsh ^= aux[3] ^ aux[11];
            r[6] = (uint32_t)hsh;
            r[7] = (uint32_t)(hsh >> 32);
        }

        ///////////////////////////////////////////////////////////////////////////
        {
            ((uint8_t*)&h2)[0] = ((uint8_t*)r)[31];
            ((uint8_t*)&h2)[1] = ((uint8_t*)r)[30];
            ((uint8_t*)&h2)[2] = ((uint8_t*)r)[29];
            ((uint8_t*)&h2)[3] = ((uint8_t*)r)[28];
            ((uint8_t*)&h2)[4] = ((uint8_t*)r)[27];
            ((uint8_t*)&h2)[5] = ((uint8_t*)r)[26];
            ((uint8_t*)&h2)[6] = ((uint8_t*)r)[25];
            ((uint8_t*)&h2)[7] = ((uint8_t*)r)[24];
        }

        ///////////////////////////////////////////////////////////////////////////
        h3 = h2 % period;

        ///////////////////////////////////////////////////////////////////////////
        #pragma unroll
        for (uint32_t i = 0; i < 8u; ++i)
        {
            r[7 - i] = be_u32(hashes[(h3 << 3) + i]);
        }

        ///////////////////////////////////////////////////////////////////////////
        B2B_INIT(aux);
        aux[0] = 0x6A09E667F2BDC928;
        ((uint64_t *)(aux))[12] ^= 71; //31+32+8;
        ((uint64_t *)(aux))[13] ^= 0;

        aux[14] = ~aux[14];

        uint8_t *bb = (uint8_t *)(&aux[16]);
        uint8_t* r1_u8 = &((uint8_t*)r)[1];

        ((uint64_t*)bb)[0] = ((uint64_t*)r1_u8)[0];
        ((uint64_t*)bb)[1] = ((uint64_t*)r1_u8)[1];
        ((uint64_t*)bb)[2] = ((uint64_t*)r1_u8)[2];
        ((uint64_t*)bb)[3] = ((uint64_t*)r1_u8)[3];

        ((uint64_t*)&bb[31])[0] = header[0];
        ((uint64_t*)&bb[39])[0] = header[1];
        ((uint64_t*)&bb[47])[0] = header[2];
        ((uint64_t*)&bb[55])[0] = header[3];

        ((uint64_t *)&bb[63])[0] = tmp;

        aux[25] = 0ull;
        aux[26] = 0ull;
        aux[27] = 0ull;
        aux[28] = 0ull;
        aux[29] = 0ull;
        aux[30] = 0ull;
        aux[31] = 0ull;

        ///////////////////////////////////////////////////////////////////////////
        devB2B_MIX(aux, aux + 16);

        ///////////////////////////////////////////////////////////////////////////
        {
            hsh = 0x6A09E667F2BDC928;
            hsh ^= aux[0] ^ aux[8];

            uint32_t const index1 = tid;
            uint32_t const index2 = 8388608u + tid;
            BHashes[index1] = be_u32((uint32_t)hsh,         0u);
            BHashes[index2] = be_u32((uint32_t)(hsh >> 32), 0u);
        }
        {
            hsh = 0xBB67AE8584CAA73B;
            hsh ^= aux[1] ^ aux[9];

            uint32_t const index1 = 16777216u + tid;
            uint32_t const index2 = 25165824u + tid;
            BHashes[index1] = be_u32((uint32_t)hsh,         0u);
            BHashes[index2] = be_u32((uint32_t)(hsh >> 32), 0u);
        }
        {
            hsh = 0x3C6EF372FE94F82B;
            hsh ^= aux[2] ^ aux[10];

            uint32_t const index1 = 33554432u + tid;
            uint32_t const index2 = 41943040u + tid;
            BHashes[index1] = be_u32((uint32_t)hsh,         0u);
            BHashes[index2] = be_u32((uint32_t)(hsh >> 32), 0u);
        }
        {
            hsh = 0xA54FF53A5F1D36F1;
            hsh ^= aux[3] ^ aux[11];

            uint32_t const index1 = 50331648u + tid;
            uint32_t const index2 = 58720256u + tid;
            BHashes[index1] = be_u32((uint32_t)hsh,         0u);
            BHashes[index2] = be_u32((uint32_t)(hsh >> 32), 0u);
        }
    }
}


__forceinline__
__device__
void update_mix(
    uint32_t* const __restrict__        shared_data,
    uint32_t* const __restrict__        shared_index,
    uint32_t const* const __restrict__  hashes,
    uint32_t const                      index_gap,
    uint32_t const                      hash_id,
    uint32_t const                      thread_id,
    uint32_t const                      thrdblck_id_by_byte,
    uint4&                              v_1,
    uint4&                              v_2)
{
    ////////////////////////////////////////////////////////////////////////////
    uint32_t const i_1 = shared_index[hash_id     ] + thread_id;
    uint32_t const i_2 = shared_index[hash_id + 8 ] + thread_id;
    uint32_t const i_3 = shared_index[hash_id + 16] + thread_id;
    uint32_t const i_4 = shared_index[hash_id + 24] + thread_id;
    uint32_t const i_5 = shared_index[hash_id + 32] + thread_id;
    uint32_t const i_6 = shared_index[hash_id + 40] + thread_id;
    uint32_t const i_7 = shared_index[hash_id + 48] + thread_id;
    uint32_t const i_8 = shared_index[hash_id + 56] + thread_id;

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const hv_1 = hashes[i_1];
    uint32_t const hv_2 = hashes[i_2];
    uint32_t const hv_3 = hashes[i_3];
    uint32_t const hv_4 = hashes[i_4];
    uint32_t const hv_5 = hashes[i_5];
    uint32_t const hv_6 = hashes[i_6];
    uint32_t const hv_7 = hashes[i_7];
    uint32_t const hv_8 = hashes[i_8];

    ////////////////////////////////////////////////////////////////////////////
    shared_data[index_gap      ] = hv_1;
    shared_data[index_gap + 64 ] = hv_2;
    shared_data[index_gap + 128] = hv_3;
    shared_data[index_gap + 192] = hv_4;
    shared_data[index_gap + 256] = hv_5;
    shared_data[index_gap + 320] = hv_6;
    shared_data[index_gap + 384] = hv_7;
    shared_data[index_gap + 448] = hv_8;

    __syncthreads();

    ////////////////////////////////////////////////////////////////////////////
    v_1.x = shared_data[thrdblck_id_by_byte + 0];
    v_1.y = shared_data[thrdblck_id_by_byte + 1];
    v_1.z = shared_data[thrdblck_id_by_byte + 2];
    v_1.w = shared_data[thrdblck_id_by_byte + 3];
    v_2.x = shared_data[thrdblck_id_by_byte + 4];
    v_2.y = shared_data[thrdblck_id_by_byte + 5];
    v_2.z = shared_data[thrdblck_id_by_byte + 6];
    v_2.w = shared_data[thrdblck_id_by_byte + 7];
}


__host__
bool autolykos_v2_v1_init(algo::hash256 const& boundary)
{
    CUDA_ER(cudaMemcpyToSymbol(d_bound, (void*)&boundary, algo::LEN_HASH_256));

    return true;
}


__global__
void __launch_bounds__(64, 64)
kernel_autolykos_v2_step_2_v1(
    uint32_t const* const __restrict__ hashes,
    uint32_t* const __restrict__       BHashes,
    volatile t_result_64* __restrict__ result,
    uint32_t const                     period,
    uint64_t const                     nonce)
{
    __shared__ uint32_t shared_index[64];
    __shared__ uint32_t shared_data[512];

    uint32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t const thread_id = threadIdx.x & 7;
    uint32_t const thrdblck_id = threadIdx.x;
    uint32_t const thrdblck_id_by_byte = thrdblck_id << 3;
    uint32_t const hash_id = threadIdx.x >> 3;
    uint32_t const hash_id_by_byte = hash_id << 3;

    uint8_t j = 0;
    uint32_t i_tmp = hash_id_by_byte + thread_id;

    uint64_t aux[32];
    uint32_t ind[32];
    uint32_t r[9];

    uint4 v1;
    uint4 v2;
    uint4 v3;
    uint4 v4;

    if (tid >= algo::autolykos_v2::NONCES_PER_ITER)
    {
        return;
    }

    r[0] = BHashes[tid           ];
    r[1] = BHashes[8388608  + tid];
    r[2] = BHashes[16777216 + tid];
    r[3] = BHashes[25165824 + tid];
    r[4] = BHashes[33554432 + tid];
    r[5] = BHashes[41943040 + tid];
    r[6] = BHashes[50331648 + tid];
    r[7] = BHashes[58720256 + tid];

    ((uint8_t* const)r)[32] = ((uint8_t const* const)r)[0];
    ((uint8_t* const)r)[33] = ((uint8_t const* const)r)[1];
    ((uint8_t* const)r)[34] = ((uint8_t const* const)r)[2];
    ((uint8_t* const)r)[35] = ((uint8_t const* const)r)[3];


    ///////////////////////////////////////////////////////////////////////////
    {
        ind[0 ] = (r[0]                          % period) << 3;
        ind[1 ] = (((r[0] << 8)  | (r[1] >> 24)) % period) << 3;
        ind[2 ] = (((r[0] << 16) | (r[1] >> 16)) % period) << 3;
        ind[3 ] = (((r[0] << 24) | (r[1] >> 8))  % period) << 3;
    }
    {
        ind[4 ] = (r[1]                          % period) << 3;
        ind[5 ] = (((r[1] << 8)  | (r[2] >> 24)) % period) << 3;
        ind[6 ] = (((r[1] << 16) | (r[2] >> 16)) % period) << 3;
        ind[7 ] = (((r[1] << 24) | (r[2] >> 8))  % period) << 3;
    }
    {
        ind[8 ] = (r[2]                          % period) << 3;
        ind[9 ] = (((r[2] << 8)  | (r[3] >> 24)) % period) << 3;
        ind[10] = (((r[2] << 16) | (r[3] >> 16)) % period) << 3;
        ind[11] = (((r[2] << 24) | (r[3] >> 8))  % period) << 3;
    }
    {
        ind[12] = (r[3]                          % period) << 3;
        ind[13] = (((r[3] << 8)  | (r[4] >> 24)) % period) << 3;
        ind[14] = (((r[3] << 16) | (r[4] >> 16)) % period) << 3;
        ind[15] = (((r[3] << 24) | (r[4] >> 8))  % period) << 3;
    }
    {
        ind[16] = (r[4]                          % period) << 3;
        ind[17] = (((r[4] << 8)  | (r[5] >> 24)) % period) << 3;
        ind[18] = (((r[4] << 16) | (r[5] >> 16)) % period) << 3;
        ind[19] = (((r[4] << 24) | (r[5] >> 8))  % period) << 3;
    }
    {
        ind[20] = (r[5]                          % period) << 3;
        ind[21] = (((r[5] << 8)  | (r[6] >> 24)) % period) << 3;
        ind[22] = (((r[5] << 16) | (r[6] >> 16)) % period) << 3;
        ind[23] = (((r[5] << 24) | (r[6] >> 8))  % period) << 3;
    }
    {
        ind[24] = (r[6]                          % period) << 3;
        ind[25] = (((r[6] << 8)  | (r[7] >> 24)) % period) << 3;
        ind[26] = (((r[6] << 16) | (r[7] >> 16)) % period) << 3;
        ind[27] = (((r[6] << 24) | (r[7] >> 8))  % period) << 3;
    }
    {
        ind[28] = (r[7]                          % period) << 3;
        ind[29] = (((r[7] << 8)  | (r[8] >> 24)) % period) << 3;
        ind[30] = (((r[7] << 16) | (r[8] >> 16)) % period) << 3;
        ind[31] = (((r[7] << 24) | (r[8] >> 8))  % period) << 3;
    }

    ///////////////////////////////////////////////////////////////////////////
    shared_index[thrdblck_id] = ind[0];
    __syncthreads();
    update_mix(shared_data,
               shared_index,
               hashes,
               i_tmp,
               hash_id,
               thread_id,
               thrdblck_id_by_byte,
               v1,
               v3);

    shared_index[thrdblck_id] = ind[1];
    __syncthreads();
    update_mix(shared_data,
               shared_index,
               hashes,
               i_tmp,
               hash_id,
               thread_id,
               thrdblck_id_by_byte,
               v2,
               v4);

    asm volatile ("add.cc.u32 %0, %1, %2;"  :"=r"(r[0]) : "r"(v1.x), "r"(v2.x));
    asm volatile ("addc.cc.u32 %0, %1, %2;" :"=r"(r[1]) : "r"(v1.y), "r"(v2.y));
    asm volatile ("addc.cc.u32 %0, %1, %2;" :"=r"(r[2]) : "r"(v1.z), "r"(v2.z));
    asm volatile ("addc.cc.u32 %0, %1, %2;" :"=r"(r[3]) : "r"(v1.w), "r"(v2.w));
    asm volatile ("addc.cc.u32 %0, %1, %2;" :"=r"(r[4]) : "r"(v3.x), "r"(v4.x));
    asm volatile ("addc.cc.u32 %0, %1, %2;" :"=r"(r[5]) : "r"(v3.y), "r"(v4.y));
    asm volatile ("addc.cc.u32 %0, %1, %2;" :"=r"(r[6]) : "r"(v3.z), "r"(v4.z));
    asm volatile ("addc.cc.u32 %0, %1, %2;" :"=r"(r[7]) : "r"(v3.w), "r"(v4.w));

    asm volatile ("addc.u32 %0, 0, 0;"      : "=r"(r[8]));

    ///////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (int k = 2; k < algo::autolykos_v2::K_LEN; ++k)
    {
        shared_index[thrdblck_id] = ind[k];
        __syncthreads();
        update_mix(shared_data,
                   shared_index,
                   hashes,
                   i_tmp,
                   hash_id,
                   thread_id,
                   thrdblck_id_by_byte,
                   v1,
                   v2);

        asm volatile ("add.cc.u32 %0, %0, %1;"  :"+r"(r[0]) : "r"(v1.x));
        asm volatile ("addc.cc.u32 %0, %0, %1;" :"+r"(r[1]) : "r"(v1.y));
        asm volatile ("addc.cc.u32 %0, %0, %1;" :"+r"(r[2]) : "r"(v1.z));
        asm volatile ("addc.cc.u32 %0, %0, %1;" :"+r"(r[3]) : "r"(v1.w));
        asm volatile ("addc.cc.u32 %0, %0, %1;" :"+r"(r[4]) : "r"(v2.x));
        asm volatile ("addc.cc.u32 %0, %0, %1;" :"+r"(r[5]) : "r"(v2.y));
        asm volatile ("addc.cc.u32 %0, %0, %1;" :"+r"(r[6]) : "r"(v2.z));
        asm volatile ("addc.cc.u32 %0, %0, %1;" :"+r"(r[7]) : "r"(v2.w));

        asm volatile ("addc.u32 %0, %0, 0;"     : "+r"(r[8]));
    }

    ////////////////////////////////////////////////////////////////////////////
    B2B_INIT(aux);
    aux[0] = 0x6A09E667F2BDC928;
    aux[12] ^= 32;
    aux[13] ^= 0;
    aux[14] = ~aux[14];

    uint8_t* bb = (uint8_t*)(&aux[16]);
    i_tmp = algo::autolykos_v2::NUM_SIZE_8 - 1;

    #pragma unroll
    for (j = 0; j < algo::autolykos_v2::NUM_SIZE_8; ++j)
    {
        bb[j] = ((uint8_t const*)r)[i_tmp - j];
    }

    aux[20] = 0;
    aux[21] = 0;
    aux[22] = 0;
    aux[23] = 0;
    aux[24] = 0;
    aux[25] = 0;
    aux[26] = 0;
    aux[27] = 0;
    aux[28] = 0;
    aux[29] = 0;
    aux[30] = 0;
    aux[31] = 0;

    ///////////////////////////////////////////////////////////////////////////
    devB2B_MIX(aux, aux + 16);

    uint64_t hsh;
    uint32_t lsb_msb[32];
    {
        hsh = 0x6A09E667F2BDC928;
        hsh ^= aux[0] ^ aux[8];
        lsb_msb[0] = (uint32_t)hsh;
        lsb_msb[1] = (uint32_t)(hsh >> 32);
    }
    {
        hsh = 0xBB67AE8584CAA73B;
        hsh ^= aux[1] ^ aux[9];
        lsb_msb[2] = (uint32_t)hsh;
        lsb_msb[3] = (uint32_t)(hsh >> 32);
    }
    {
        hsh = 0x3C6EF372FE94F82B;
        hsh ^= aux[2] ^ aux[10];
        lsb_msb[4] = (uint32_t)hsh;
        lsb_msb[5] = (uint32_t)(hsh >> 32);
    }
    {
        hsh = 0xA54FF53A5F1D36F1;
        hsh ^= aux[3] ^ aux[11];
        lsb_msb[6] = (uint32_t)hsh;
        lsb_msb[7] = (uint32_t)(hsh >> 32);
    }


    ///////////////////////////////////////////////////////////////////////////
    i_tmp = algo::autolykos_v2::NUM_SIZE_8 - 1;
    #pragma unroll
    for (j = 0; j < algo::autolykos_v2::NUM_SIZE_8; j++)
    {
        ((uint8_t*)r)[j] = ((uint8_t*)lsb_msb)[i_tmp - j];
    }


    ///////////////////////////////////////////////////////////////////////////
    uint64_t const* const r64 = (uint64_t const* const)r;
    uint64_t const* const bound64 = (uint64_t const* const)d_bound;


    uint64_t const r3 = r64[3];
    uint64_t const r2 = r64[2];
    uint64_t const r1 = r64[1];
    uint64_t const r0 = r64[0];

    uint64_t const b3 = bound64[3];
    uint64_t const b2 = bound64[2];
    uint64_t const b1 = bound64[1];
    uint64_t const b0 = bound64[0];

    j =    ((r0 < b0) && (r1 == b1))
        || ((r1 < b1) && (r2 == b2))
        || ((r2 < b2) && (r3 == b3))
        || (r3 < b3);

    if (j)
    {
        uint32_t const index = atomicAdd((uint32_t*)&result->index, 1);
        if (index < MAX_RESULT_INDEX)
        {
            result->found = true;
            result->nonce[index] = tid + nonce;
        }
    }
}


__host__
bool autolykos_v2_v1(
    cudaStream_t stream,
    t_result_64* result,
    uint32_t* dag,
    uint32_t* header,
    uint32_t* BHashes,
    uint32_t const blocks,
    uint32_t const threads,
    uint32_t const period)
{
    uint64_t const nonce{ 11055774138563218679ull };

    kernel_autolykos_v2_step1_v1<<<blocks / 4u, threads, 0, stream>>>
    (
        dag,
        (uint64_t*)header,
        BHashes,
        nonce,
        period
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    kernel_autolykos_v2_step_2_v1<<<blocks, threads, 0, stream>>>
    (
        dag,
        BHashes,
        result,
        period,
        nonce
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
