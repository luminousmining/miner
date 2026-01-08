#pragma once

////////////////////////////////////////////////////////////////////////////////
#include <string>

////////////////////////////////////////////////////////////////////////////////
#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <benchmark/result.hpp>

#if defined(CUDA_ENABLE)
////////////////////////////////////////////////////////////////////////////////
bool init_array(cudaStream_t stream,
                uint32_t* const dest,
                uint64_t const size);

////////////////////////////////////////////////////////////////////////////////
bool init_ethash_ethminer(algo::hash1024 const* dagHash,
                          algo::hash256 const* headerHash,
                          uint64_t const dagNumberItem,
                          uint64_t const boundary);
bool ethash_ethminer(cudaStream_t stream,
                     t_result_64* result,
                     uint32_t const blocks,
                     uint32_t const threads);

////////////////////////////////////////////////////////////////////////////////
bool autolykos_v2_mhssamadi_init(algo::hash256 const& boundary);
bool autolykos_v2_mhssamadi_prehash(cudaStream_t stream,
                                    uint32_t* hashes,
                                    uint32_t const blocks,
                                    uint32_t const threads,
                                    uint32_t const period,
                                    uint32_t const height);
bool autolykos_v2_mhssamadi(cudaStream_t stream,
                            t_result_64* result,
                            uint32_t const* dag,
                            uint32_t* BHashes,
                            uint32_t* header,
                            uint32_t const blocks,
                            uint32_t const threads,
                            uint32_t const period,
                            uint32_t const height);

////////////////////////////////////////////////////////////////////////////////
bool autolykos_v2_init_lm1(algo::hash256 const& boundary);
bool autolykos_v2_prehash_lm1(cudaStream_t stream,
                              uint32_t* dag,
                              uint32_t const blocks,
                              uint32_t const threads,
                              uint32_t const period,
                              uint32_t const height);
bool autolykos_v2_lm1(cudaStream_t stream,
                      t_result_64* result,
                      uint32_t* dag,
                      uint32_t* header,
                      uint32_t* BHashes,
                      uint32_t const blocks,
                      uint32_t const threads,
                      uint32_t const period);
bool autolykos_v2_init_lm2(algo::hash256 const& boundary);
bool autolykos_v2_lm2(cudaStream_t stream,
                      t_result_64* result,
                      uint32_t* dag,
                      uint32_t* header,
                      uint32_t* BHashes,
                      uint32_t const blocks,
                      uint32_t const threads,
                      uint32_t const period);

////////////////////////////////////////////////////////////////////////////////
#define PARAMETER_KAWPOW cudaStream_t stream,                                  \
                         t_result* result,                                     \
                         uint32_t* const header,                               \
                         uint32_t* const dag,                                  \
                         uint32_t const blocks,                                \
                         uint32_t const threads


bool kawpow_kawpowminer_1(PARAMETER_KAWPOW);
bool kawpow_kawpowminer_2(PARAMETER_KAWPOW);
bool kawpow_lm1(PARAMETER_KAWPOW);
bool kawpow_lm2(PARAMETER_KAWPOW);
bool kawpow_lm3(PARAMETER_KAWPOW);
bool kawpow_lm4(PARAMETER_KAWPOW);
bool kawpow_lm5(PARAMETER_KAWPOW);
bool kawpow_lm6(PARAMETER_KAWPOW);
bool kawpow_lm7(PARAMETER_KAWPOW);
bool kawpow_lm8(PARAMETER_KAWPOW);
bool kawpow_lm9(PARAMETER_KAWPOW);
bool kawpow_lm10(PARAMETER_KAWPOW);


////////////////////////////////////////////////////////////////////////////////
#define PARAMETER_KECCAK_F800 cudaStream_t stream,                             \
                         uint32_t const blocks,                                \
                         uint32_t const threads

bool keccak_f800_lm1(PARAMETER_KECCAK_F800);
bool keccak_f800_lm2(PARAMETER_KECCAK_F800);
bool keccak_f800_lm3(PARAMETER_KECCAK_F800);
bool keccak_f800_lm4(PARAMETER_KECCAK_F800);
bool keccak_f800_lm5(PARAMETER_KECCAK_F800);
bool keccak_f800_lm6(PARAMETER_KECCAK_F800);
bool keccak_f800_lm7(PARAMETER_KECCAK_F800);
bool keccak_f800_lm8(PARAMETER_KECCAK_F800);
bool keccak_f800_lm9(PARAMETER_KECCAK_F800);

#endif
