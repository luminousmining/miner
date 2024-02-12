#pragma once


#include <cstdint>


namespace algo
{
    namespace kawpow
    {
        // Override Ethash
        constexpr uint32_t DAG_ITEM_PARENTS { 512u };

        // Override ProgPOW
        constexpr uint32_t MAX_PERIOD{ 3u };
        constexpr uint32_t COUNT_CACHE{ 11u };
        constexpr uint32_t COUNT_MATH{ 18u };
    }
}
