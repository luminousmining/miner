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
uint32_t bswap32_d(uint32_t const x)
{
    return (x >> 24) | ((x >> 8) & 0x0000FF00u) | ((x << 8) & 0x00FF0000u) | (x << 24);
}


// NOTE: ported to match the AMD/OpenCL kernel (sources/algo/blake3/opencl/blake3.cl),
// which is POCL- and RX 9070 XT-verified against an independent BLAKE3 oracle and accepted
// live by woolypooly + HeroMiners. Alephium PoW = BLAKE3(BLAKE3(nonce(24) || headerBlob(302))):
// the 24-byte nonce (big-endian 8-byte value + 16 zero bytes) is PREPENDED. The old kernel
// overwrote header[0..7] with a little-endian nonce in a SHARED buffer (a per-thread race) and
// hashed the wrong layout. This port is compile-checked but not runtime-verified (no NVIDIA GPU).
__global__
void kernel_blake3_search(
    algo::blake3::Result* __restrict__ result,
    uint32_t const* __restrict__ const header,
    uint32_t const* __restrict__ const target,
    uint64_t const startNonce,
    uint32_t const fromGroup,
    uint32_t const toGroup)
{
    uint32_t const thread_id{ ((blockIdx.x * blockDim.x) + threadIdx.x) };
    uint64_t const nonce{ thread_id + startNonce };

    // Private per-thread buffer (no shared race): buf = nonce(24) || headerBlob(302).
    uint32_t buffer[82];
    #pragma unroll
    for (uint32_t i{ 0u }; i < 82u; ++i)
    {
        buffer[i] = 0u;
    }
    buffer[0] = bswap32_d(static_cast<uint32_t>(nonce >> 32));
    buffer[1] = bswap32_d(static_cast<uint32_t>(nonce & 0xFFFFFFFFu));
    // buffer[2..5] = 0  (nonce bytes 8..23)
    #pragma unroll
    for (uint32_t i{ 0u }; i < 76u; ++i)
    {
        buffer[6u + i] = header[i];   // headerBlob shifted up by 24 bytes (6 words)
    }

    ////////////////////////////////////////////////////////////////////////////
    // Pass 1: one chunk over 326 bytes = CHUNK_START + 4 empty + 6-byte END|ROOT.
    uint32_t vector[algo::blake3::VECTOR_LENGTH];
    initialize_vector(vector);
    blake3_compress_pre(vector, &buffer[0], algo::blake3::BLOCK_LENGTH, algo::blake3::FLAG_CHUNK_START);
    #pragma unroll
    for (uint32_t i{ 0u }; i < algo::blake3::CHUNK_LOOP_HEADER; ++i)
    {
        blake3_compress_pre(vector, &buffer[16u * (i + 1u)], algo::blake3::BLOCK_LENGTH, algo::blake3::FLAG_EMPTY);
    }
    uint32_t last[16]{};
    last[0] = buffer[80];
    last[1] = buffer[81] & 0x0000FFFFu;
    blake3_compress_pre(vector, last, 6u, algo::blake3::FLAG_END_AND_ROOT);

    ////////////////////////////////////////////////////////////////////////////
    // Pass 2: BLAKE3 of the 32-byte chaining value (single CHUNK_START|END|ROOT block).
    uint32_t hash[algo::blake3::HASH_LENGTH]{};
    #pragma unroll
    for (uint32_t i{ 0u }; i < algo::blake3::VECTOR_LENGTH; ++i)
    {
        hash[i] = vector[i];
    }
    initialize_vector(vector);
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
