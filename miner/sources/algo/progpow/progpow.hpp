#pragma once

#if !defined(__LIB_CUDA)
#include <cstdint>
#include <string>
#include <sstream>

#include <algo/crypto/kiss99.hpp>
#endif


namespace algo
{
    namespace progpow
    {
        enum class VERSION : uint8_t
        {
            V_0_9_2,
            V_0_9_3,
            KAWPOW,
            FIROPOW,
            EVRPROGPOW
        };

        constexpr uint32_t LANES{ 16u };
        constexpr uint32_t REGS{ 32u };
        constexpr uint32_t DAG_LOADS{ 4u };
        constexpr uint32_t CACHE_BYTES{ 16u * 1024u };
        constexpr uint32_t COUNT_DAG{ 64u };
        constexpr uint32_t EPOCH_LENGTH{ 7500u };
        constexpr uint32_t MODULE_SOURCE{ algo::progpow::REGS * (algo::progpow::REGS - 1u) };
        constexpr uint32_t MODULE_CACHE{ algo::progpow::CACHE_BYTES / sizeof(uint32_t) };

        namespace v_0_9_2
        {
            constexpr uint32_t MAX_PERIOD{ 50u };
            constexpr uint32_t COUNT_CACHE{ 12u };
            constexpr uint32_t COUNT_MATH{ 20u };
        }

        namespace v_0_9_3
        {
            constexpr uint32_t MAX_PERIOD{ 10u };
            constexpr uint32_t COUNT_CACHE{ 11u };
            constexpr uint32_t COUNT_MATH{ 18u };
        }

#if !defined(__LIB_CUDA)
        algo::Kiss99Properties initializeRound(uint64_t const period,
                                               int32_t* const dst,
                                               int32_t* const src);
        void writeMathRandomKernelCuda(VERSION const progpowVersion,
                                       uint32_t const deviceId,
                                       uint64_t const period,
                                       uint32_t const countCache,
                                       uint32_t const countMath);
        void writeMathRandomKernelOpenCL(VERSION const progpowVersion,
                                         uint32_t const deviceId,
                                         uint64_t const period,
                                         uint32_t const countCache,
                                         uint32_t const countMath);

        namespace nvidia
        {
            void writeSequenceMergeEntries(std::stringstream& ss,
                                           uint32_t const i,
                                           uint32_t const x,
                                           uint32_t const sel);
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
            void writeSequenceMergeEntries(std::stringstream& ss,
                                           uint32_t const i,
                                           uint32_t const x,
                                           uint32_t const sel);
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
#endif
    }
}
