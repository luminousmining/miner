#pragma once

#include <cstring>
#include <sstream>

#include <algo/hash.hpp>

namespace algo
{
    namespace meowpow
    {
        // Override Ethash
        constexpr uint32_t DAG_ITEM_PARENTS { 512u };
        constexpr uint32_t EPOCH_LENGTH{ 7500u };
        constexpr uint32_t MAX_EPOCH_NUMBER{ 110u };

        // Override ProgPOW
        constexpr uint32_t REGS{ 16u };
        constexpr uint32_t MAX_PERIOD{ 6u };
        constexpr uint32_t COUNT_CACHE{ 6u };
        constexpr uint32_t COUNT_MATH{ 9u };
    }
}
