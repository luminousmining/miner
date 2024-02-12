__global__
void autolykos_fill_dag(
    uint32_t* const __restrict__ hashes,
    uint32_t  const period,
    uint32_t  const height)
{
    uint32_t const tid{ threadIdx.x + (blockDim.x * blockIdx.x) };
    if (tid >= period)
    {
        return;
    }

    //====================================================================//
    //  Initialize context
    //====================================================================//
    uint64_t h[8]
    {
        0x6A09E667F3BCC908UL,
        0xBB67AE8584CAA73BUL,
        0x3C6EF372FE94F82BUL,
        0xA54FF53A5F1D36F1UL,
        0x510E527FADE682D1UL,
        0x9B05688C2B3E6C1FUL,
        0x1F83D9ABFB41BD6BUL,
        0x5BE0CD19137E2179UL
    };
    uint64_t b[16];
    uint64_t t = 0;

    h[0] ^= 0x01010020;

    //====================================================================//
    //  Hash tid
    //====================================================================//
    ((uint32_t *)b)[0] = __byte_perm(tid, tid, 0x0123);

    //====================================================================//
    //  Hash height
    //====================================================================//
    ((uint32_t *)b)[1] = height;

    //====================================================================//
    //  Hash constant message
    //====================================================================//
    uint64_t ctr = 0;
    for (int x = 1; x < 16; ++x, ++ctr)
    {
        b[x] = be_u64(ctr);
    }

    #pragma unroll 1
    for (uint32_t z = 0; z < 63u; ++z)
    {
        t += 128;
        blake2b((uint64_t *)h, (uint64_t *)b, t, 0UL);

        #pragma unroll
        for (uint32_t x = 0; x < 16u; ++x, ++ctr)
        {
            b[x] = be_u64(ctr);
        }
    }

    t += 128;
    blake2b((uint64_t *)h, (uint64_t *)b, t, 0UL);

    b[0] = be_u64(ctr);

    t += 8;

    #pragma unroll
    for (int i = 1; i < 16; ++i)
    {
        ((uint64_t *)b)[i] = 0UL;
    }

    blake2b((uint64_t *)h, (uint64_t *)b, t, 0xFFFFFFFFFFFFFFFFUL);

    //====================================================================//
    //  Dump result to global memory -- BIG ENDIAN
    //====================================================================//

    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        ((uint64_t *)hashes)[(tid + 1) * 4 - i - 1] = be_u64(h[i]);
    }

    ((uint8_t *)hashes)[tid * 32 + 31] = 0;
}


__host__
bool autolykosv2BuildDag(
    cudaStream_t stream,
    resolver::nvidia::autolykos_v2::KernelParameters& params)
{
    autolykos_fill_dag<<<params.hostDagItemCount / 64, 64, 0, stream>>>
    (
        (uint32_t*)params.dag,
        params.hostPeriod,
        params.hostHeight
    );
    CUDA_ER(cudaStreamSynchronize(stream));
    CUDA_ER(cudaGetLastError());

    return true;
}
