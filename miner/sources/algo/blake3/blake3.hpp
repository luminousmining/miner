#pragma once

#include <cstdint>

#include <algo/hash.hpp>

namespace algo
{
    namespace blake3
    {
        ////////////////////////////////////////////////////////////////////////
        constexpr uint32_t KEY_LENGTH        { 32u                                 };
        constexpr uint32_t OUT_LENGTH        { 32u                                 };
        constexpr uint32_t CHUNK_LENGTH      { 1024u                               };
        constexpr uint32_t HEADER_U8_CAP     { 384u                                };
        constexpr uint32_t HEADER_U32_CAP    { algo::blake3::HEADER_U8_CAP / 4u    };
        constexpr uint32_t HEADER_U8_LENGTH  { 326u                                };
        constexpr uint32_t HEADER_U32_LENGTH { algo::blake3::HEADER_U8_LENGTH / 4u };
        constexpr uint32_t VECTOR_LENGTH     { 8u                                  };
        constexpr uint32_t BLOCK_LENGTH      { 64u                                 };
        constexpr uint32_t HASH_LENGTH       { 16u                                 };

        ////////////////////////////////////////////////////////////////////////
        constexpr uint8_t FLAG_EMPTY        { 0  };
        constexpr uint8_t FLAG_CHUNK_START  { 1  }; // 1 << 0
        constexpr uint8_t FLAG_CHUNK_END    { 2  }; // 1 << 1
        constexpr uint8_t FLAG_ROOT         { 8  }; // 1 << 2
        constexpr uint8_t FLAG_END_AND_ROOT { 10 }; // (1 << 2) | (1 << 1)

        ////////////////////////////////////////////////////////////////////////
        constexpr uint32_t CHUNK_LOOP_HEADER { 4u };

        ////////////////////////////////////////////////////////////////////////
        constexpr uint32_t VECTOR_INDEX_0 { 0x6A09E667u };
        constexpr uint32_t VECTOR_INDEX_1 { 0xBB67AE85u };
        constexpr uint32_t VECTOR_INDEX_2 { 0x3C6EF372u };
        constexpr uint32_t VECTOR_INDEX_3 { 0xA54FF53Au };
        constexpr uint32_t VECTOR_INDEX_4 { 0x510E527Fu };
        constexpr uint32_t VECTOR_INDEX_5 { 0x9B05688Cu };
        constexpr uint32_t VECTOR_INDEX_6 { 0x1F83D9ABu };
        constexpr uint32_t VECTOR_INDEX_7 { 0x5BE0CD19u };

        ////////////////////////////////////////////////////////////////////////
        constexpr uint32_t CHAIN_NUMBER { 16u };
        constexpr uint32_t GROUP_NUMBER { 4u  };

    }
}
