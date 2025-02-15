#pragma once

#include <cstdint>
#include <sstream>

#if !defined(__LIB_CUDA)
#include <algo/hash.hpp>
#include <algo/crypto/kiss99.hpp>
#endif


namespace algo
{
    namespace progpow_quai
    {
        // Override Ethash
        constexpr uint32_t DAG_ITEM_PARENTS{ 512u };
        constexpr uint32_t EPOCH_LENGTH{ 2147483647u };
        // constexpr uint64_t LIGHT_CACHE_GROWTH{ 1 << 21 };
        // constexpr uint64_t LIGHT_CACHE_COUNT_ITEMS_GROWTH{ algo::progpow_quai::LIGHT_CACHE_GROWTH / algo::LEN_HASH_512 };
        // constexpr uint32_t DAG_INIT_SIZE{ 1 << 32 };

        // Override ProgPOW
        constexpr uint32_t MAX_PERIOD{ 2147483647u };
        constexpr uint32_t COUNT_CACHE{ 11u };
        constexpr uint32_t COUNT_MATH{ 18u };

        namespace nvidia
        {
            void writeSequenceMathMerge(std::stringstream& ss,
                                        uint32_t const i,
                                        uint32_t const dst,
                                        uint32_t const src1,
                                        uint32_t const src2,
                                        uint32_t const sel_math,
                                        uint32_t const sel_merge);
            void writeSequenceMergeCache(std::stringstream& ss,
                                         uint32_t const i,
                                         uint32_t const src,
                                         uint32_t const dst,
                                         uint32_t const sel);
        }

        namespace amd
        {
            void writeSequenceMathMerge(std::stringstream& ss,
                                        uint32_t const i,
                                        uint32_t const dst,
                                        uint32_t const src1,
                                        uint32_t const src2,
                                        uint32_t const sel_math,
                                        uint32_t const sel_merge);
            void writeSequenceMergeCache(std::stringstream& ss,
                                         uint32_t const i,
                                         uint32_t const src,
                                         uint32_t const dst,
                                         uint32_t const sel);
        }
    }
}
