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
//    printf("=====================TURN[0]=======================\n");
    blake3_compress_pre(
        vector,
        header,
        algo::blake3::BLOCK_LENGTH,
        algo::blake3::FLAG_CHUNK_START);
//    THD_PRINT_BUFFER("final vector", vector, algo::blake3::VECTOR_LENGTH);
    header += 16;
//    printf("============================================\n");

    ////////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (uint32_t i{ 0u }; i < algo::blake3::CHUNK_LOOP_HEADER; ++i)
    {
//        printf("=====================TURN[%u]=======================\n", i + 1u);
        blake3_compress_pre(
            vector,
            header,
            algo::blake3::BLOCK_LENGTH,
            algo::blake3::FLAG_EMPTY);
//        THD_PRINT_BUFFER("final vector", vector, algo::blake3::VECTOR_LENGTH);
        header += 16;
//        printf("============================================\n");
    }

    ////////////////////////////////////////////////////////////////////////////
//    printf("=====================TURN[5]=======================\n");
    memset((uint8_t*)header + 6u, 0, 58);
    blake3_compress_pre(
        vector,
        header,
        6u,
        algo::blake3::FLAG_END_AND_ROOT);
//    THD_PRINT_BUFFER("final vector", vector, algo::blake3::VECTOR_LENGTH);
//    printf("============================================\n");
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

//    printf("*****************************************************\n");
//    THD_PRINT_BUFFER("header", buffer, algo::LEN_HASH_3072_WORD_32);
//    THD_PRINT_BUFFER("target", target, algo::LEN_HASH_256_WORD_32);
//    printf("*****************************************************\n");

    ////////////////////////////////////////////////////////////////////////////
    initialize_vector(vector);
    chunk_header(vector, buffer);
//    THD_PRINT_BUFFER("chunk_header", vector, algo::blake3::VECTOR_LENGTH);
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

    ////////////////////////////////////////////////////////////////////////////
//    THD_PRINT_BUFFER("final hash", hash,   algo::blake3::HASH_LENGTH);

    if (true == isLowerOrEqual((uint8_t*)hash, (uint8_t*)target, algo::LEN_HASH_256_WORD_8))
    {
//        printf("check_target => true - tid[%llu]\n", nonce);
        uint32_t const bigIndex{ (hash[7] >> 24) % algo::blake3::CHAIN_NUMBER };
        if (   (bigIndex / algo::blake3::GROUP_NUMBER) == fromGroup
            && (bigIndex % algo::blake3::GROUP_NUMBER) == toGroup)
        {
//            printf("check_index => true - tid[%llu]\n", nonce);
            uint32_t const index = atomicAdd((uint32_t*)&result->count, 1);
            if (index < algo::blake3::MAX_RESULT)
            {
//                printf("====> [check_hash valid] <====\n");
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
    uint32_t const blocks,
    uint32_t const threads)
{
    kernel_blake3_search<<<blocks, threads, 0, stream>>>(
        params.resultCache,
        params.header->word32,
        params.target->word32,
        params.hostNonce,
        params.hostFromGroup,
        params.hostToGroup);
}
