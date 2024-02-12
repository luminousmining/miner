#pragma once

#include <cstring>

#include <algo/hash.hpp>


namespace algo
{
    namespace firopow
    {
        // Override Ethash
        constexpr uint32_t DAG_ITEM_PARENTS { 512u };
        constexpr uint32_t EPOCH_LENGTH{ 1300u };
        constexpr uint32_t DAG_INIT_SIZE{ (1 << 30) + (1 << 29) };
        constexpr uint64_t DAG_COUNT_ITEMS_INIT{ algo::firopow::DAG_INIT_SIZE / algo::LEN_HASH_1024 };

        // Override ProgPOW
        constexpr uint32_t MAX_PERIOD{ 1u };
        constexpr uint32_t COUNT_CACHE{ 11u };
        constexpr uint32_t COUNT_MATH{ 18u };
    }
}