#pragma once

#include <cstring>

#include <algo/hash.hpp>


namespace algo
{
    namespace evrprogpow
    {
        // Override Ethash
        constexpr uint32_t DAG_ITEM_PARENTS { 512u };
        constexpr uint32_t EPOCH_LENGTH{ 12000u };
        constexpr uint32_t DAG_INIT_SIZE{ (1u << 30u) * 3u };
        constexpr uint64_t DAG_COUNT_ITEMS_INIT{ DAG_INIT_SIZE / LEN_HASH_1024 };

        // Override ProgPOW
        constexpr uint32_t MAX_PERIOD{ 3u };
        constexpr uint32_t COUNT_CACHE{ 11u };
        constexpr uint32_t COUNT_MATH{ 18u };
    }
}