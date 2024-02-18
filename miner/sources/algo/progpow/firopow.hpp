#pragma once

#include <cstring>
#include <sstream>

#include <algo/hash.hpp>


namespace algo
{
    namespace firopow
    {
        // Override Ethash
        constexpr uint32_t DAG_ITEM_PARENTS { 512u };
        constexpr uint32_t EPOCH_LENGTH{ 1300u };
        constexpr uint32_t DAG_INIT_SIZE{ (1 << 30) + (1 << 29) };
        constexpr uint64_t DAG_COUNT_ITEMS_INIT{ DAG_INIT_SIZE / LEN_HASH_1024 };

        // Override ProgPOW
        constexpr uint32_t MAX_PERIOD{ 1u };
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