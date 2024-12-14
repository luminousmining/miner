#pragma once

#include <cstdint>

#include <algo/hash.hpp>

namespace algo
{
    namespace autolykos_v2
    {
        constexpr uint32_t BLOCK_BEGIN { 0x96000  }; // 614400
        constexpr uint32_t BLOCK_END   { 0x401000 }; // 4198400

        constexpr uint32_t EPOCH_MIN { 0x4000000  }; // 4194304
        constexpr uint32_t EPOCH_MAX { 0x7FC9FF98 }; // 2143944600

        constexpr uint32_t EPOCH_PERIOD { 0xC800 }; // 51200

        constexpr uint32_t NONCES_PER_ITER { 8388608u };
        constexpr uint32_t NONCES_PER_THREAD { 1u };
        constexpr uint32_t THREADS_PER_ITER { NONCES_PER_ITER / NONCES_PER_THREAD };

        constexpr uint32_t NONCE_SIZE_8 { sizeof(uint64_t) };
        constexpr uint32_t NONCE_SIZE_32 { NONCE_SIZE_8 >> 2 };

        constexpr uint32_t NUM_SIZE_8 { algo::LEN_HASH_256 };
        constexpr uint32_t NUM_SIZE_32 { NUM_SIZE_8 >> 2 };

        constexpr uint32_t BUF_SIZE_8 { 128u };
        constexpr uint32_t K_LEN { 32u };
        constexpr uint32_t MAX_SOLS { 16u };

        //  TODO : A revoir pour AMD
        constexpr uint32_t AMD_BLOCK_DIM { 64u };
        constexpr uint32_t AMD_NONCES_PER_ITER { 0x400000 }; // 4194304
        constexpr uint32_t AMD_NONCES_PER_THREAD { 1u };
        constexpr uint32_t AMD_THREADS_PER_ITER { (AMD_NONCES_PER_ITER / AMD_NONCES_PER_THREAD) };

#if !defined(__LIB_CUDA)
        uint32_t computePeriod(uint32_t const blockNumner);
#endif //!__LIB_CUDA
    }
}
