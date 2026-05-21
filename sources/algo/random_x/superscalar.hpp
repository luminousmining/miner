#pragma once

#include <cstdint>


namespace algo
{
    namespace random_x
    {
        // SuperscalarHash instruction types (subset of RandomX VM for dataset building)
        enum class ScalarInstType : uint8_t
        {
            ISUB_R   = 0,  // dst -= src         (latency 1)
            IXOR_R   = 1,  // dst ^= src         (latency 1)
            IADD_RS  = 2,  // dst += src << shift (latency 1)
            IMUL_R   = 3,  // dst *= src         (latency 3, MUL port)
            IROR_C   = 4,  // dst = ror(dst, imm) (latency 1)
            IADD_C   = 5,  // dst += imm32       (latency 1)
            IXOR_C   = 6,  // dst ^= imm32       (latency 1)
            IMULH_R  = 7,  // dst = (dst*src)>>64 unsigned (latency 4, 2 uops)
            ISMULH_R = 8,  // dst = (dst*src)>>64 signed   (latency 4, 2 uops)
            IMUL_RCP = 9,  // dst *= rcp(imm32)  (latency 3, 2 uops)
        };

        static constexpr uint32_t SUPERSCALAR_MAX_INSTRUCTIONS{ 512u };
        static constexpr uint32_t SUPERSCALAR_LATENCY         { 170u };
        static constexpr uint32_t SUPERSCALAR_ITERS           { 8u };

        struct ScalarInst
        {
            ScalarInstType type{ ScalarInstType::ISUB_R };
            uint8_t        dst { 0u };
            uint8_t        src { 0u };
            uint32_t       imm { 0u };
        };

        struct SuperscalarProgram
        {
            ScalarInst instructions[SUPERSCALAR_MAX_INSTRUCTIONS]{};
            uint32_t   size{ 0u };
            uint32_t   addressReg{ 0u };  // register with max dependency chain → cache index
        };

        // Build 8 SuperscalarHash programs from the cache key (32-byte seed_hash).
        // One BlakeGenerator is shared across all 8 programs.
        void buildSuperscalarPrograms(
            uint8_t const*      cacheKey,
            SuperscalarProgram  programs[SUPERSCALAR_ITERS]);

        // Execute one SuperscalarHash program on the 8 integer registers.
        void executeSuperscalarProgram(
            SuperscalarProgram const& prog,
            uint64_t                  r[8]);

        // Compute floor(2^x / divisor) for IMUL_RCP — exposed for GPU upload.
        // Returns 0 for divisor == 0 or power-of-2 (no-op cases).
        uint64_t superscalarComputeReciprocal(uint32_t divisor);
    }
}
