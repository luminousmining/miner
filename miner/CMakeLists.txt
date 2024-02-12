################################################################################
###  LIB CUDA ETHASH                                                        ####
################################################################################
cuda_add_library(${LIB_CRYPTO_NVIDIA} STATIC
    sources/algo/autolykos/cuda/autolykos_v2.cu
    sources/algo/ethash/cuda/ethash.cu
    sources/algo/progpow/cuda/progpow.cu)

target_compile_definitions(${LIB_CRYPTO_NVIDIA} PRIVATE __LIB_CUDA)

include_directories(${LIB_CRYPTO_NVIDIA} PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/sources
)

################################################################################
### HEADERS AND SOURCES FILES MINER                                         ####
################################################################################
set(MINER_HEADERS)
set(MINER_SOURCES
sources/miner.cpp
)


################################################################################
###  LINK MINER                                                             ####
################################################################################
add_executable(${MINER_EXE})
add_subdirectory(sources)

if (NOT WIN32)
    target_link_options(${MINER_EXE} PUBLIC -static-libstdc++ -static-libgcc)
endif()

target_sources(${MINER_EXE} PUBLIC
    ${MINER_HEADERS}
    ${MINER_SOURCES}
)

include_directories(${MINER_EXE} PUBLIC
    ${Boost_INCLUDE_DIR}
    ${OPENSSL_INCLUDE_DIR}
    ${OpenCL_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/sources
)

target_link_libraries(${MINER_EXE}
    ${GNUTLS_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OPEN_SSL_LIBRARIES}
    ${OpenCL_LIBRARIES}
    ${CUDA_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIBRARY}
    ${CUDA_NVPTX_LIBRARY}
    ${CUDA_NVRTC_BUILTINS_LIBRARY}
    ${LIB_CRYPTO_NVIDIA}
)


################################################################################
### COMMON FILES OPENCL                                                     ####
################################################################################
set(IN_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/sources/common/opencl)
set(OUT_COMMON ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/common)

file(COPY ${IN_COMMON}/rotate_byte.cl DESTINATION ${OUT_COMMON})
file(COPY ${IN_COMMON}/to_u4.cl DESTINATION ${OUT_COMMON})
file(COPY ${IN_COMMON}/xor.cl DESTINATION ${OUT_COMMON})

################################################################################
### CRYPTO FILES OPENCL                                                     ####
################################################################################
set(IN_CRYPTO ${CMAKE_CURRENT_SOURCE_DIR}/sources/algo/crypto/opencl)
set(OUT_CRYPTO ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/crypto)

file(COPY ${IN_CRYPTO}/blake2b.cl DESTINATION ${OUT_CRYPTO})
file(COPY ${IN_CRYPTO}/blake2b_compress.cl DESTINATION ${OUT_CRYPTO})
file(COPY ${IN_CRYPTO}/fnv1.cl DESTINATION ${OUT_CRYPTO})
file(COPY ${IN_CRYPTO}/keccak_f1600.cl DESTINATION ${OUT_CRYPTO})
file(COPY ${IN_CRYPTO}/keccak_f800.cl DESTINATION ${OUT_CRYPTO})
file(COPY ${IN_CRYPTO}/kiss99.cl DESTINATION ${OUT_CRYPTO})


################################################################################
### ETHASH KERNEL OPENCL                                                    ####
################################################################################
set(IN_ETHASH ${CMAKE_CURRENT_SOURCE_DIR}/sources/algo/ethash/opencl)
set(OUT_ETHASH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/ethash)

file(COPY ${IN_ETHASH}/ethash_dag.cl DESTINATION ${OUT_ETHASH})
file(COPY ${IN_ETHASH}/ethash_result.cl DESTINATION ${OUT_ETHASH})
file(COPY ${IN_ETHASH}/ethash_search.cl DESTINATION ${OUT_ETHASH})

################################################################################
### COMMON FILES CUDA                                                       ####
################################################################################
set(IN_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/sources/common/cuda)
set(OUT_COMMON ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/common)

file(COPY ${IN_COMMON}/be_u32.cuh DESTINATION ${OUT_COMMON})
file(COPY ${IN_COMMON}/be_u64.cuh DESTINATION ${OUT_COMMON})
file(COPY ${IN_COMMON}/copy_u4.cuh DESTINATION ${OUT_COMMON})
file(COPY ${IN_COMMON}/get_lane_id.cuh DESTINATION ${OUT_COMMON})
file(COPY ${IN_COMMON}/rotate_byte.cuh DESTINATION ${OUT_COMMON})
file(COPY ${IN_COMMON}/to_u4.cuh DESTINATION ${OUT_COMMON})
file(COPY ${IN_COMMON}/to_u64.cuh DESTINATION ${OUT_COMMON})
file(COPY ${IN_COMMON}/xor.cuh DESTINATION ${OUT_COMMON})


################################################################################
### CRYPTO FILES CUDA                                                       ####
################################################################################
set(IN_CRYPTO ${CMAKE_CURRENT_SOURCE_DIR}/sources/algo/crypto/cuda)
set(OUT_CRYPTO ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/crypto)

file(COPY ${IN_CRYPTO}/fnv1.cuh DESTINATION ${OUT_CRYPTO})
file(COPY ${IN_CRYPTO}/keccak_f800.cuh DESTINATION ${OUT_CRYPTO})
file(COPY ${IN_CRYPTO}/kiss99.cuh DESTINATION ${OUT_CRYPTO})

################################################################################
### PROGPOW KERNEL CUDA                                                     ####
################################################################################
set(IN_PROGPOW ${CMAKE_CURRENT_SOURCE_DIR}/sources/algo/progpow/cuda)
set(OUT_PROGPOW ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/progpow)

file(COPY sources/algo/progpow/result.hpp DESTINATION ${OUT_PROGPOW})

file(COPY ${IN_PROGPOW}/firopow_seed.cuh DESTINATION ${OUT_PROGPOW})
file(COPY ${IN_PROGPOW}/progpow_seed.cuh DESTINATION ${OUT_PROGPOW})
file(COPY ${IN_PROGPOW}/kawpow_seed.cuh DESTINATION ${OUT_PROGPOW})
file(COPY ${IN_PROGPOW}/search.cu DESTINATION ${OUT_PROGPOW})

################################################################################
### PROGPOW KERNEL OPENCL                                                   ####
################################################################################
set(IN_PROGPOW ${CMAKE_CURRENT_SOURCE_DIR}/sources/algo/progpow/opencl)
set(OUT_PROGPOW ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/progpow)

file(COPY ${IN_PROGPOW}/progpow_result.cl DESTINATION ${OUT_PROGPOW})
file(COPY ${IN_PROGPOW}/firopow_seed.cl DESTINATION ${OUT_PROGPOW})
file(COPY ${IN_PROGPOW}/progpow_seed.cl DESTINATION ${OUT_PROGPOW})
file(COPY ${IN_PROGPOW}/kawpow_seed.cl DESTINATION ${OUT_PROGPOW})
file(COPY ${IN_PROGPOW}/progpow.cl DESTINATION ${OUT_PROGPOW})

################################################################################
### AUTOLYKOS V2 KERNEL OPENCL                                              ####
################################################################################
set(IN_AUTOLYKOS_V2 ${CMAKE_CURRENT_SOURCE_DIR}/sources/algo/autolykos/opencl)
set(OUT_AUTOLYKOS_V2 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/kernel/autolykos)

file(COPY ${IN_AUTOLYKOS_V2}/autolykos_v2_dag.cl DESTINATION ${OUT_AUTOLYKOS_V2})
file(COPY ${IN_AUTOLYKOS_V2}/autolykos_v2_result.cl DESTINATION ${OUT_AUTOLYKOS_V2})
file(COPY ${IN_AUTOLYKOS_V2}/autolykos_v2_search.cl DESTINATION ${OUT_AUTOLYKOS_V2})
file(COPY ${IN_AUTOLYKOS_V2}/autolykos_v2_var_global.cl DESTINATION ${OUT_AUTOLYKOS_V2})
file(COPY ${IN_AUTOLYKOS_V2}/autolykos_v2_verify.cl DESTINATION ${OUT_AUTOLYKOS_V2})
