#pragma once

#include <algo/hash.hpp>


constexpr uint32_t MAX_RESULT_INDEX{ 4u };


struct t_result_32
{
    bool          error{ false };
    bool          found{ false };
    std::uint32_t index{ 0u };
    std::uint64_t nonce[MAX_RESULT_INDEX]{ 0u, };
    std::uint32_t mix[MAX_RESULT_INDEX][algo::LEN_HASH_256_WORD_32]{{ 0u, }, };
};


struct t_result_64
{
    bool          error{ false };
    bool          found{ false };
    std::uint32_t index{ 0u };
    std::uint64_t nonce[MAX_RESULT_INDEX]{ 0ull, };
    std::uint64_t mix[MAX_RESULT_INDEX][algo::LEN_HASH_256_WORD_64]{ {0ull, }, };
};
