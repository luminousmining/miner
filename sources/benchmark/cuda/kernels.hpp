#pragma once

///////////////////////////////////////////////////////////////////////////////
#include <string>

///////////////////////////////////////////////////////////////////////////////
#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <benchmark/result.hpp>

///////////////////////////////////////////////////////////////////////////////
bool init_array(cudaStream_t stream,
                uint32_t* const dest,
                uint64_t const size);

///////////////////////////////////////////////////////////////////////////////
bool init_ethash_ethminer(
    algo::hash1024 const* dagHash,
    algo::hash256 const* headerHash,
    uint64_t const dagNumberItem,
    uint64_t const boundary);
bool ethash_ethminer(cudaStream_t stream,
               t_result_64* result,
               uint32_t const blocks,
               uint32_t const threads);

