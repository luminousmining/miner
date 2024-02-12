#pragma once

#define DAG_PARRALLEL_LANE 4u
#define DAG_HASH_U4_SIZE 4u


__device__ __forceinline__
void doCopy(
    uint4* __restrict__ const dst,
    uint4* __restrict__ const src,
    uint32_t const workerId,
    uint32_t const laneId)
{
    ////////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (uint32_t i{ 0u }; i < DAG_HASH_U4_SIZE; ++i)
    {
        uint32_t const x{  __shfl_sync(0xffffffff, src[i].x, laneId, DAG_PARRALLEL_LANE) };
        uint32_t const y{  __shfl_sync(0xffffffff, src[i].y, laneId, DAG_PARRALLEL_LANE) };
        uint32_t const z{  __shfl_sync(0xffffffff, src[i].z, laneId, DAG_PARRALLEL_LANE) };
        uint32_t const w{  __shfl_sync(0xffffffff, src[i].w, laneId, DAG_PARRALLEL_LANE) };

        if (i == workerId)
        {
            dst->x = x;
            dst->y = y;
            dst->z = z;
            dst->w = w;
        }
    }
}


__device__ __forceinline__
void buildItemFromCache(
    uint4* const __restrict__ hash,
    uint32_t const workerId,
    uint32_t const cacheIndex)
{
    ////////////////////////////////////////////////////////////////////////////
    uint32_t const cacheIndexLimit{ (cacheIndex % d_light_number_item) * DAG_HASH_U4_SIZE };

    ////////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (uint32_t laneId{ 0u }; laneId < DAG_PARRALLEL_LANE; ++laneId)
    {
        ////////////////////////////////////////////////////////////////////////
        uint32_t const tmpCacheIndex{ __shfl_sync(0xffffffff, cacheIndexLimit, laneId, DAG_PARRALLEL_LANE) };
        uint32_t const index{ tmpCacheIndex + workerId };
        uint4 cache16Bytes = d_light_cache[index];

        ////////////////////////////////////////////////////////////////////////
        uint4 cache64Bytes[DAG_HASH_U4_SIZE];
        #pragma unroll
        for (uint32_t i{ 0u }; i < DAG_HASH_U4_SIZE; ++i)
        {
            cache64Bytes[i].x = __shfl_sync(0xffffffff, cache16Bytes.x, i, DAG_PARRALLEL_LANE);
            cache64Bytes[i].y = __shfl_sync(0xffffffff, cache16Bytes.y, i, DAG_PARRALLEL_LANE);
            cache64Bytes[i].z = __shfl_sync(0xffffffff, cache16Bytes.z, i, DAG_PARRALLEL_LANE);
            cache64Bytes[i].w = __shfl_sync(0xffffffff, cache16Bytes.w, i, DAG_PARRALLEL_LANE);
        }

        ////////////////////////////////////////////////////////////////////////
        if (laneId == workerId)
        {
            #pragma unroll
            for (uint32_t i{ 0u }; i < DAG_HASH_U4_SIZE; ++i)
            {
                fnv1(&hash[i], &cache64Bytes[i]);
            }
        }
    }
}


__device__ __forceinline__
void buildItem(
    bool is_access,
    uint32_t const workerId,
    uint32_t const dagIndex,
    uint32_t const loopIndex,
    uint32_t const cacheIndex)
{
    ////////////////////////////////////////////////////////////////////////////
    uint4 hash[DAG_HASH_U4_SIZE];
    uint32_t const cacheStartIndex{ (cacheIndex % d_light_number_item) * DAG_HASH_U4_SIZE };

    ////////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (uint32_t lane_id{ 0u }; lane_id < DAG_PARRALLEL_LANE; ++lane_id)
    {
        ////////////////////////////////////////////////////////////////////////
        uint32_t const tmpCacheStartIndex{ __shfl_sync(0xffffffff, cacheStartIndex, lane_id, DAG_PARRALLEL_LANE) };
        uint32_t const tmpCacheIndex{ tmpCacheStartIndex + workerId };
        uint4 const tmp_cache{ d_light_cache[tmpCacheIndex] };
        uint4 tmp_hash[DAG_HASH_U4_SIZE];

        ////////////////////////////////////////////////////////////////////////
        #pragma unroll
        for (uint32_t i{ 0u }; i < DAG_PARRALLEL_LANE; ++i)
        {
            tmp_hash[i].x = __shfl_sync(0xffffffff, tmp_cache.x, i, DAG_PARRALLEL_LANE);
            tmp_hash[i].y = __shfl_sync(0xffffffff, tmp_cache.y, i, DAG_PARRALLEL_LANE);
            tmp_hash[i].z = __shfl_sync(0xffffffff, tmp_cache.z, i, DAG_PARRALLEL_LANE);
            tmp_hash[i].w = __shfl_sync(0xffffffff, tmp_cache.w, i, DAG_PARRALLEL_LANE);
        }

        ////////////////////////////////////////////////////////////////////////
        if (lane_id == workerId)
        {
            #pragma unroll
            for (uint32_t i{ 0u }; i < DAG_HASH_U4_SIZE; ++i)
            {
                hash[i].x = tmp_hash[i].x;
                hash[i].y = tmp_hash[i].y;
                hash[i].z = tmp_hash[i].z;
                hash[i].w = tmp_hash[i].w;
            }
        }
    }
    hash[0].x ^= cacheIndex;

    ////////////////////////////////////////////////////////////////////////////
    keccak_f1600(hash);

    ////////////////////////////////////////////////////////////////////////////
    uint32_t j{ 0u };
    #pragma unroll
    for (uint32_t y{ 0u }; y < loopIndex; ++y)
    {
        #pragma unroll
        for (uint32_t i{ 0u }; i < DAG_HASH_U4_SIZE; ++i)
        {
            buildItemFromCache(hash, workerId, fnv1(cacheIndex ^ j,        hash[i].x));
            buildItemFromCache(hash, workerId, fnv1(cacheIndex ^ (j + 1u), hash[i].y));
            buildItemFromCache(hash, workerId, fnv1(cacheIndex ^ (j + 2u), hash[i].z));
            buildItemFromCache(hash, workerId, fnv1(cacheIndex ^ (j + 3u), hash[i].w));

            j += 4u;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    keccak_f1600(hash);

    ////////////////////////////////////////////////////////////////////////////
    uint4 itemHash;
    #pragma unroll
    for (uint32_t laneId{ 0u }; laneId < DAG_HASH_U4_SIZE; ++laneId)
    {
        doCopy(&itemHash, hash, workerId, laneId);
        uint32_t const tmpDagIndex{ __shfl_sync(0xffffffff, dagIndex, laneId, DAG_PARRALLEL_LANE) };
        if (true == is_access)
        {
            uint32_t const index{ tmpDagIndex + workerId };
            d_dag[index] = itemHash;
        }
    }
}


__global__
void kernelProgpowBuildDag(
    uint32_t const startIndex,
    uint32_t const loop)
{
    ////////////////////////////////////////////////////////////////////////////
    uint32_t const threadId{ (blockIdx.x * blockDim.x) + threadIdx.x };
    uint32_t const workerId{ threadId & 3u };
    uint32_t const dagAccessWorker{ (threadId / 4u) + startIndex };
    uint32_t const dagAccessIndex{ threadId + startIndex };
    uint32_t const dagIndex{ threadId + startIndex };

    ////////////////////////////////////////////////////////////////////////////
    if (dagAccessWorker >= d_dag_number_item)
    {
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const cacheIndex{ dagIndex * 2u };
    uint32_t const dagIndex1{ dagIndex * 8u };
    uint32_t const dagIndex2{ dagIndex1 + 4u };
    bool const isAccess{ dagAccessIndex < d_dag_number_item };
    buildItem(isAccess, workerId, dagIndex1, loop, cacheIndex);
    buildItem(isAccess, workerId, dagIndex2, loop, cacheIndex + 1u);
}


__host__
bool progpowBuildDag(
    cudaStream_t stream,
    uint32_t const dagItemParents,
    uint32_t const dagNumberItems)
{
    uint32_t const itemByKernel{ 65536u };
    uint32_t const loop{ dagItemParents / 4u / 4u };

    for (uint32_t i{ 0u }; i < dagNumberItems; i += itemByKernel)
    {
        kernelProgpowBuildDag<<<512, 128, 0, stream>>>(i, loop);
        CUDA_ER(cudaStreamSynchronize(stream));
        CUDA_ER(cudaGetLastError());
    }

    return true;
}
