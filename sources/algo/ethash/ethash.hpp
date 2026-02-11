#pragma once

#if !defined(__LIB_CUDA)
#include <algo/hash.hpp>
#include <algo/dag_context.hpp>
#endif


namespace algo
{
    namespace ethash
    {
        constexpr uint32_t REVISION{ 23u };
        constexpr uint32_t EPOCH_LENGTH{ 30000u };
        constexpr uint32_t NUM_DAG_ACCESSES{ 64u };
        constexpr uint32_t MAX_EPOCH_NUMBER{ 32639u };

        constexpr uint32_t EIP1099_EPOCH_LENGTH{ 60000u };
        constexpr uint32_t EIP1099_MAX_EPOCH_NUMBER{ 30000u };
        constexpr uint32_t EIP1057_MAX_EPOCH_NUMER{ 110u };

        constexpr uint64_t LIGHT_CACHE_INIT_SIZE{ 1u << 24 };
        constexpr uint64_t LIGHT_CACHE_GROWTH{ 1u << 17 };
        constexpr uint64_t LIGHT_CACHE_ROUNDS{ 3ull };
        constexpr uint64_t LIGHT_CACHE_COUNT_ITEMS_INIT{ algo::ethash::LIGHT_CACHE_INIT_SIZE / algo::LEN_HASH_512 };
        constexpr uint64_t LIGHT_CACHE_COUNT_ITEMS_GROWTH{ algo::ethash::LIGHT_CACHE_GROWTH / algo::LEN_HASH_512 };

        constexpr uint32_t DAG_INIT_SIZE{ 1u << 30 };
        constexpr uint32_t DAG_GROWTH{ 1u << 23 };
        constexpr uint32_t DAG_ITEM_PARENTS{ 256u };
        constexpr uint64_t DAG_COUNT_ITEMS_INIT{ algo::ethash::DAG_INIT_SIZE / algo::LEN_HASH_1024 };
        constexpr uint64_t DAG_COUNT_ITEMS_GROWTH{ algo::ethash::DAG_GROWTH / algo::LEN_HASH_1024 };

#if !defined(__LIB_CUDA)
        int32_t findEpoch(algo::hash256 const& seedHash,
                          uint32_t const maxEpoch);
        void initializeDagContext(algo::DagContext& context,
                                  uint64_t const currentEpoch,
                                  uint32_t const maxEpoch,
                                  uint64_t const dagCountItemsGrowth,
                                  uint64_t const dagCountItemsInit,
                                  uint32_t const lightCacheCountItemsGrowth,
                                  uint32_t const lightCacheCountItemsInit);
        void freeDagContext(algo::DagContext& context);
        void buildLightCache(algo::DagContext& context);
#endif
    }
}
