#pragma once

#define DAG_PARRALLEL_LANE 4u
#define DAG_HASH_U4_SIZE 4u


__device__ __forceinline__
uint4 doCopy(
    uint4* __restrict__ const src,
    uint32_t const worker_id,
    uint32_t const lane_id)
{
    ////////////////////////////////////////////////////////////////////////////
    uint4 item;

    ////////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (uint32_t i = 0u; i < DAG_HASH_U4_SIZE; ++i)
    {
        uint4 const tmp_src = src[i];
        uint32_t const x = reg_load(tmp_src.x, lane_id, DAG_PARRALLEL_LANE);
        uint32_t const y = reg_load(tmp_src.y, lane_id, DAG_PARRALLEL_LANE);
        uint32_t const z = reg_load(tmp_src.z, lane_id, DAG_PARRALLEL_LANE);
        uint32_t const w = reg_load(tmp_src.w, lane_id, DAG_PARRALLEL_LANE);

        if (i == worker_id)
        {
            item.x = x;
            item.y = y;
            item.z = z;
            item.w = w;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    return item;
}


__device__ __forceinline__
void buildItemFromCache(
    uint4* const __restrict__ hash,
    uint32_t const worker_id,
    uint32_t const cache_index)
{
    ////////////////////////////////////////////////////////////////////////////
    uint32_t const cache_index_limit = (cache_index % d_light_number_item) * DAG_HASH_U4_SIZE;

    ////////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (uint32_t lane_id = 0u; lane_id < DAG_PARRALLEL_LANE; ++lane_id)
    {
        ////////////////////////////////////////////////////////////////////////
        uint32_t const tmp_cache_index = reg_load(cache_index_limit, lane_id, DAG_PARRALLEL_LANE);
        uint32_t const index = tmp_cache_index + worker_id;
        uint4 const cache16Bytes = d_light_cache[index];

        ////////////////////////////////////////////////////////////////////////
        uint4 cache64Bytes[DAG_HASH_U4_SIZE];
        #pragma unroll
        for (uint32_t i = 0u; i < DAG_HASH_U4_SIZE; ++i)
        {
            cache64Bytes[i].x = reg_load(cache16Bytes.x, i, DAG_PARRALLEL_LANE);
            cache64Bytes[i].y = reg_load(cache16Bytes.y, i, DAG_PARRALLEL_LANE);
            cache64Bytes[i].z = reg_load(cache16Bytes.z, i, DAG_PARRALLEL_LANE);
            cache64Bytes[i].w = reg_load(cache16Bytes.w, i, DAG_PARRALLEL_LANE);
        }

        ////////////////////////////////////////////////////////////////////////
        if (lane_id == worker_id)
        {
            #pragma unroll
            for (uint32_t i = 0u; i < DAG_HASH_U4_SIZE; ++i)
            {
                fnv1(&hash[i], &cache64Bytes[i]);
            }
        }
    }
}


__device__ __forceinline__
void buildItem(
    bool const is_access,
    uint32_t const worker_id,
    uint32_t const dag_index,
    uint32_t const loop_index,
    uint32_t const cache_index)
{
    ////////////////////////////////////////////////////////////////////////////
    uint4 hash[DAG_HASH_U4_SIZE];
    uint32_t const cache_start_index = (cache_index % d_light_number_item) * DAG_HASH_U4_SIZE;

    ////////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (uint32_t lane_id = 0u; lane_id < DAG_PARRALLEL_LANE; ++lane_id)
    {
        ////////////////////////////////////////////////////////////////////////
        uint32_t const tmp_cache_start_index = reg_load(cache_start_index, lane_id, DAG_PARRALLEL_LANE);
        uint32_t const tmp_cache_index = tmp_cache_start_index + worker_id;
        uint4 const tmp_cache = d_light_cache[tmp_cache_index];
        uint4 tmp_hash[DAG_HASH_U4_SIZE];

        ////////////////////////////////////////////////////////////////////////
        #pragma unroll
        for (uint32_t i = 0u; i < DAG_PARRALLEL_LANE; ++i)
        {
            tmp_hash[i].x = reg_load(tmp_cache.x, i, DAG_PARRALLEL_LANE);
            tmp_hash[i].y = reg_load(tmp_cache.y, i, DAG_PARRALLEL_LANE);
            tmp_hash[i].z = reg_load(tmp_cache.z, i, DAG_PARRALLEL_LANE);
            tmp_hash[i].w = reg_load(tmp_cache.w, i, DAG_PARRALLEL_LANE);
        }

        ////////////////////////////////////////////////////////////////////////
        if (lane_id == worker_id)
        {
            #pragma unroll
            for (uint32_t i = 0u; i < DAG_HASH_U4_SIZE; ++i)
            {
                hash[i].x = tmp_hash[i].x;
                hash[i].y = tmp_hash[i].y;
                hash[i].z = tmp_hash[i].z;
                hash[i].w = tmp_hash[i].w;
            }
        }
    }
    hash[0].x ^= cache_index;

    ////////////////////////////////////////////////////////////////////////////
    keccak_f1600(hash);

    ////////////////////////////////////////////////////////////////////////////
    uint32_t j = 0u;
    #pragma unroll
    for (uint32_t y = 0u; y < loop_index; ++y)
    {
        #pragma unroll
        for (uint32_t i = 0u; i < DAG_HASH_U4_SIZE; ++i)
        {
            buildItemFromCache(hash, worker_id, fnv1(cache_index ^ j,        hash[i].x));
            buildItemFromCache(hash, worker_id, fnv1(cache_index ^ (j + 1u), hash[i].y));
            buildItemFromCache(hash, worker_id, fnv1(cache_index ^ (j + 2u), hash[i].z));
            buildItemFromCache(hash, worker_id, fnv1(cache_index ^ (j + 3u), hash[i].w));

            j += 4u;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    keccak_f1600(hash);

    ////////////////////////////////////////////////////////////////////////////
    #pragma unroll
    for (uint32_t lane_id = 0u; lane_id < DAG_HASH_U4_SIZE; ++lane_id)
    {
        uint4 const itemHash = doCopy(hash, worker_id, lane_id);
        uint32_t const tmpdag_index = reg_load(dag_index, lane_id, DAG_PARRALLEL_LANE);
        if (true == is_access)
        {
            uint32_t const index = tmpdag_index + worker_id;
            d_dag[index] = itemHash;
        }
    }
}


__global__
void kernelProgpowBuildDag(
    uint32_t const start_index,
    uint32_t const loop)
{
    ////////////////////////////////////////////////////////////////////////////
    uint32_t const threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t const worker_id = threadId & 3u;
    uint32_t const dagAccessWorker = (threadId / 4u) + start_index;
    uint32_t const dag_index = threadId + start_index;

    ////////////////////////////////////////////////////////////////////////////
    if (dagAccessWorker >= d_dag_number_item)
    {
        return;
    }

    ////////////////////////////////////////////////////////////////////////////
    uint32_t const cache_index = dag_index * 2u;
    uint32_t const dag_index_1 = dag_index * 8u;
    uint32_t const dag_index_2 = dag_index_1 + 4u;
    bool const isAccess = dag_index < d_dag_number_item;
    buildItem(isAccess, worker_id, dag_index_1, loop, cache_index);
    buildItem(isAccess, worker_id, dag_index_2, loop, cache_index + 1u);
}


__host__
bool progpowBuildDag(
    cudaStream_t stream,
    uint32_t const dagItemParents,
    uint32_t const dagNumberItems)
{
    uint32_t const threads = 128u;
    uint32_t const blocks = 512u;
    uint32_t const itemByKernel = threads * blocks;
    uint32_t const loop = dagItemParents / DAG_PARRALLEL_LANE / DAG_HASH_U4_SIZE;

    for (uint32_t i = 0u; i < dagNumberItems; i += itemByKernel)
    {
        kernelProgpowBuildDag<<<blocks, threads, 0, stream>>>(i, loop);
        CUDA_ER(cudaStreamSynchronize(stream));
        CUDA_ER(cudaGetLastError());
    }

    return true;
}
