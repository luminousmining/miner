#pragma once

#include <cstdint>
#include <sstream>


namespace algo
{
    namespace meowpow
    {
        // Override Ethash
        constexpr uint32_t DAG_ITEM_PARENTS { 512u };

        // Override ProgPOW
        constexpr uint32_t REGS{ 16u };
        constexpr uint32_t MODULE_SOURCE{ algo::meowpow::REGS * (algo::meowpow::REGS - 1u) };
        constexpr uint32_t MAX_PERIOD{ 6u };
        constexpr uint32_t COUNT_CACHE{ 6u };
        constexpr uint32_t COUNT_MATH{ 9u };

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
