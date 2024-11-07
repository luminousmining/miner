#pragma once

struct t_result
{
    bool          error;
    bool          found;
    std::uint32_t index;
    std::uint64_t nonce[4];
    std::uint32_t mix[4][algo::LEN_HASH_256_WORD_32];
};


struct t_result_64
{
    bool          error;
    bool          found;
    std::uint32_t index;
    std::uint64_t nonce[4];
    std::uint64_t mix[4][algo::LEN_HASH_256_WORD_64];
};
