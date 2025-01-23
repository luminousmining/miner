#pragma once


__forceinline__ __device__
void initialize_vector(
    uint32_t* const vector)
{
    vector[0] = 0x6A09E667u;
    vector[1] = 0xBB67AE85u;
    vector[2] = 0x3C6EF372u;
    vector[3] = 0xA54FF53Au;
    vector[4] = 0x510E527Fu;
    vector[5] = 0x9B05688Cu;
    vector[6] = 0x1F83D9ABu;
    vector[7] = 0x5BE0CD19u;
}


__forceinline__ __device__
void chunk_header(
    uint32_t* __restrict__ const vector,
    uint32_t* __restrict__ header)
{
    ////////////////////////////////////////////////////////////////////////////
    blake3_compress_pre(
        vector,
        header,
        algo::blake3::BLOCK_LENGTH,
        algo::blake3::FLAG_CHUNK_START);
    header += 16;

    ////////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (uint32_t i{ 0u }; i < algo::blake3::CHUNK_LOOP_HEADER; ++i)
    {
        blake3_compress_pre(
            vector,
            header,
            algo::blake3::BLOCK_LENGTH,
            algo::blake3::FLAG_EMPTY);
        header += 16;
    }

    ////////////////////////////////////////////////////////////////////////////
    memset((uint8_t*)header + 6u, 0, 58);
    blake3_compress_pre(
        vector,
        header,
        6u,
        algo::blake3::FLAG_END_AND_ROOT);
}


__global__
void kernel_blake3_search(
    algo::blake3::Result* __restrict__ result,
    uint32_t* __restrict__ const header,
    uint32_t* __restrict__ const target,
    uint64_t const startNonce,
    uint32_t const fromGroup,
    uint32_t const toGroup)
{
    __shared__ uint32_t buffer[algo::LEN_HASH_3072_WORD_32];
    uint32_t vector[algo::blake3::VECTOR_LENGTH];
    uint32_t hash[algo::blake3::HASH_LENGTH];

    uint32_t const thread_id{ ((blockIdx.x * blockDim.x) + threadIdx.x) };
    uint64_t const nonce{ thread_id + startNonce };

    if (threadIdx.x < algo::LEN_HASH_3072_WORD_32)
    {
        buffer[threadIdx.x] = header[threadIdx.x];
    }
    __syncthreads();
    ((uint64_t*)buffer)[0] = nonce;

    ////////////////////////////////////////////////////////////////////////////
    initialize_vector(vector);
    chunk_header(vector, buffer);
    #pragma unroll
    for (uint32_t i{ 0u }; i < algo::blake3::VECTOR_LENGTH; ++i)
    {
        hash[i] = vector[i];
    }
    #pragma unroll
    for (uint32_t i{ algo::blake3::VECTOR_LENGTH }; i < algo::blake3::HASH_LENGTH; ++i)
    {
        hash[i] = 0u;
    }

    ////////////////////////////////////////////////////////////////////////////
    initialize_vector(vector);
    memset((uint8_t*)hash + 32, 0, 32);
    blake3_compress_pre(vector, hash, algo::blake3::BLOCK_LENGTH / 2u, 11u);
    #pragma unroll
    for (uint32_t i{ 0u }; i < algo::blake3::VECTOR_LENGTH; ++i)
    {
        hash[i] = vector[i];
    }


    if (true == isLowerOrEqual((uint8_t*)hash, (uint8_t*)target, algo::LEN_HASH_256_WORD_8))
    {
        uint32_t const bigIndex{ (hash[7] >> 24) % algo::blake3::CHAIN_NUMBER };
        if (   (bigIndex / algo::blake3::GROUP_NUMBER) == fromGroup
            && (bigIndex % algo::blake3::GROUP_NUMBER) == toGroup)
        {
            uint32_t const index = atomicAdd((uint32_t*)&result->count, 1);
            if (index < algo::blake3::MAX_RESULT)
            {
                result->found = true;
                result->nonces[index] = nonce;
            }
        }
    }
}


__host__
void blake3Search(
    cudaStream_t stream,
    resolver::nvidia::blake3::KernelParameters& params,
    uint32_t const currentIndexStream,
    uint32_t const blocks,
    uint32_t const threads)
{
    kernel_blake3_search<<<blocks, threads, 0, stream>>>(
        &params.resultCache[currentIndexStream],
        params.header->word32,
        params.target->word32,
        params.hostNonce,
        params.hostFromGroup,
        params.hostToGroup);
}
