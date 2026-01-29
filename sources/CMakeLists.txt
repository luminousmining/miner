################################################################################
###  LIB CUDA ETHASH                                                        ####
################################################################################
if (BUILD_NVIDIA)
    cuda_add_library(${LIB_CRYPTO_NVIDIA} STATIC
        algo/autolykos/cuda/autolykos_v2.cu
        algo/blake3/cuda/blake3.cu
        algo/ethash/cuda/ethash.cu
        algo/progpow/cuda/progpow.cu)

    target_compile_definitions(${LIB_CRYPTO_NVIDIA} PRIVATE __LIB_CUDA CUDA_ENABLE)

    include_directories(${LIB_CRYPTO_NVIDIA} PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

################################################################################
###  LIB CUDA BENCH                                                        ####
################################################################################
if (BUILD_NVIDIA)
    cuda_add_library(${LIB_BENCH_NVIDIA} STATIC
        benchmark/cuda/autolykos_v2/autolykos_v2_mhssamadani_prehash.cu
        benchmark/cuda/autolykos_v2/autolykos_v2_mhssamadani.cu
        benchmark/cuda/autolykos_v2/autolykos_v2_prehash_v1.cu
        benchmark/cuda/autolykos_v2/autolykos_v2_lm1.cu
        benchmark/cuda/autolykos_v2/autolykos_v2_lm2.cu
        benchmark/cuda/common/array.cu
        benchmark/cuda/ethash/ethash_base.cu
        benchmark/cuda/ethash/ethash_ethminer.cu
        benchmark/cuda/ethash/ethash_lm1.cu
        benchmark/cuda/ethash/ethash_lm2.cu
        benchmark/cuda/ethash_light_cache/ethash_light_cache_lm1.cu
        benchmark/cuda/ethash_light_cache/ethash_light_cache_lm2.cu
        benchmark/cuda/fnv1/fnv1_lm1.cu
        benchmark/cuda/fnv1/fnv1_lm2.cu
        benchmark/cuda/kawpow/kawpow_kawpowminer_1.cu
        benchmark/cuda/kawpow/kawpow_kawpowminer_2.cu
        benchmark/cuda/kawpow/kawpow_lm1.cu
        benchmark/cuda/kawpow/kawpow_lm2.cu
        benchmark/cuda/kawpow/kawpow_lm3.cu
        benchmark/cuda/kawpow/kawpow_lm4.cu
        benchmark/cuda/kawpow/kawpow_lm5.cu
        benchmark/cuda/kawpow/kawpow_lm6.cu
        benchmark/cuda/kawpow/kawpow_lm7.cu
        benchmark/cuda/kawpow/kawpow_lm8.cu
        benchmark/cuda/kawpow/kawpow_lm9.cu
        benchmark/cuda/kawpow/kawpow_lm10.cu
        benchmark/cuda/keccak/keccak_f800_lm1.cu
        benchmark/cuda/keccak/keccak_f800_lm2.cu
        benchmark/cuda/keccak/keccak_f800_lm3.cu
        benchmark/cuda/keccak/keccak_f800_lm4.cu
        benchmark/cuda/keccak/keccak_f800_lm5.cu
        benchmark/cuda/keccak/keccak_f800_lm6.cu
        benchmark/cuda/keccak/keccak_f800_lm7.cu
        benchmark/cuda/keccak/keccak_f800_lm8.cu
        benchmark/cuda/keccak/keccak_f800_lm9.cu
    )

    target_compile_definitions(${LIB_BENCH_NVIDIA} PRIVATE __LIB_CUDA CUDA_ENABLE)

    include_directories(${LIB_BENCH_NVIDIA} PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()


################################################################################
### HEADERS AND SOURCES FILES MINER                                         ####
################################################################################
set(MINER_HEADERS)
set(MINER_SOURCES
miner.cpp
)

################################################################################
### HEADERS AND SOURCES FILES BENCHMARK                                     ####
################################################################################
set(BENCH_HEADERS)
set(BENCH_SOURCES
benchmark.cpp
)

################################################################################
### HEADERS AND SOURCES FILES UNIT TEST                                     ####
################################################################################
set(UNIT_TEST_HEADERS)
set(UNIT_TEST_SOURCES
unit_test.cpp
)

################################################################################
## LINK UNIT TEST                                                           ####
################################################################################
# Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(GoogleTest)

add_executable(${UNIT_TEST_EXE})
#gtest_discover_tests(${UNIT_TEST_EXE})

################################################################################
###  APP MINER                                                              ####
################################################################################
add_executable(${MINER_EXE})

################################################################################
###  APP BENCHMARK                                                          ####
################################################################################
add_executable(${BENCH_EXE})

################################################################################
###  SOURCES                                                                ####
################################################################################
add_subdirectory(algo)
add_subdirectory(api)
add_subdirectory(benchmark)
add_subdirectory(common)
add_subdirectory(device)
add_subdirectory(network)
add_subdirectory(profiler)
add_subdirectory(resolver)
add_subdirectory(statistical)
add_subdirectory(stratum)

set(SOURCES_ALGO ${SOURCES_ALGO} PARENT_SCOPE)
set(SOURCES_ALGO_AUTOLYOS ${SOURCES_ALGO_AUTOLYOS} PARENT_SCOPE)
set(SOURCES_ALGO_BLAKE3 ${SOURCES_ALGO_BLAKE3} PARENT_SCOPE)
set(SOURCES_ALGO_CRYPTO ${SOURCES_ALGO_CRYPTO} PARENT_SCOPE)
set(SOURCES_ALGO_CRYPTO_TESTS ${SOURCES_ALGO_CRYPTO_TESTS} PARENT_SCOPE)
set(SOURCES_ALGO_ETHASH ${SOURCES_ALGO_ETHASH} PARENT_SCOPE)
set(SOURCES_ALGO_ETHASH_TESTS ${SOURCES_ALGO_ETHASH_TESTS} PARENT_SCOPE)
set(SOURCES_ALGO_PROGPOW ${SOURCES_ALGO_PROGPOW} PARENT_SCOPE)
set(SOURCES_ALGO_TESTS ${SOURCES_ALGO_TESTS} PARENT_SCOPE)
set(SOURCES_API ${SOURCES_API} PARENT_SCOPE)
set(SOURCES_BENCHMARK ${SOURCES_BENCHMARK} PARENT_SCOPE)
set(SOURCES_COMMON ${SOURCES_COMMON} PARENT_SCOPE)
set(SOURCES_COMMON_CLI ${SOURCES_COMMON_CLI} PARENT_SCOPE)
set(SOURCES_COMMON_ERROR ${SOURCES_COMMON_ERROR} PARENT_SCOPE)
set(SOURCES_COMMON_KERNEL_GENERATOR ${SOURCES_COMMON_KERNEL_GENERATOR} PARENT_SCOPE)
set(SOURCES_COMMON_LOG ${SOURCES_COMMON_LOG} PARENT_SCOPE)
set(SOURCES_COMMON_MOCKER ${SOURCES_COMMON_MOCKER} PARENT_SCOPE)
set(SOURCES_COMMON_TOOL ${SOURCES_COMMON_TOOL} PARENT_SCOPE)
set(SOURCES_DEVICE ${SOURCES_DEVICE} PARENT_SCOPE)
set(SOURCES_NETWORK ${SOURCES_NETWORK} PARENT_SCOPE)
set(SOURCES_PROFILER ${SOURCES_PROFILER} PARENT_SCOPE)
set(SOURCES_PROFILER_ADL ${SOURCES_PROFILER_ADL} PARENT_SCOPE)
set(SOURCES_RESOLVER ${SOURCES_RESOLVER} PARENT_SCOPE)
set(SOURCES_RESOLVER_AMD ${SOURCES_RESOLVER_AMD} PARENT_SCOPE)
set(SOURCES_RESOLVER_AMD_TESTS ${SOURCES_RESOLVER_AMD_TESTS} PARENT_SCOPE)
set(SOURCES_RESOLVER_CPU ${SOURCES_RESOLVER_CPU} PARENT_SCOPE)
set(SOURCES_RESOLVER_CPU_TESTS ${SOURCES_RESOLVER_CPU_TESTS} PARENT_SCOPE)
set(SOURCES_RESOLVER_NVIDIA ${SOURCES_RESOLVER_NVIDIA} PARENT_SCOPE)
set(SOURCES_RESOLVER_NVIDIA_TESTS ${SOURCES_RESOLVER_NVIDIA_TESTS} PARENT_SCOPE)
set(SOURCES_STRATISTICAL ${SOURCES_STRATISTICAL} PARENT_SCOPE)
set(SOURCES_STRATUM ${SOURCES_STRATUM} PARENT_SCOPE)

################################################################################
### FILTER SOLUTION VISUAL STUDIO                                            ###
################################################################################
source_group("" FILES benchmark.cpp miner.cpp unit_test.cpp)
source_group("algo" FILES ${SOURCES_ALGO})
source_group("algo/autolykos" FILES ${SOURCES_ALGO_AUTOLYOS})
source_group("algo/blake3" FILES ${SOURCES_ALGO_BLAKE3})
source_group("algo/crypto" FILES ${SOURCES_ALGO_CRYPTO})
source_group("algo/crypto/tests" FILES ${SOURCES_ALGO_CRYPTO_TESTS})
source_group("algo/ethash" FILES ${SOURCES_ALGO_ETHASH})
source_group("algo/ethash/tests" FILES ${SOURCES_ALGO_ETHASH_TESTS})
source_group("algo/progpow" FILES ${SOURCES_ALGO_PROGPOW})
source_group("algo/tests" FILES ${SOURCES_ALGO_TESTS})
source_group("api" FILES ${SOURCES_API})
source_group("benchmark" FILES ${SOURCES_BENCHMARK})
source_group("common" FILES ${SOURCES_COMMON})
source_group("common/cli" FILES ${SOURCES_COMMON_CLI})
source_group("common/error" FILES ${SOURCES_COMMON_ERROR})
source_group("common/kernel_generator" FILES ${SOURCES_COMMON_KERNEL_GENERATOR})
source_group("common/log" FILES ${SOURCES_COMMON_LOG})
source_group("common/mocker" FILES ${SOURCES_COMMON_MOCKER})
source_group("common/tool" FILES ${SOURCES_COMMON_TOOL})
source_group("device" FILES ${SOURCES_DEVICE})
source_group("network" FILES ${SOURCES_NETWORK})
source_group("profiler" FILES ${SOURCES_PROFILER})
source_group("profiler/adl" FILES ${SOURCES_PROFILER_ADL})
source_group("resolver" FILES ${SOURCES_RESOLVER})
source_group("resolver/amd" FILES ${SOURCES_RESOLVER_AMD})
source_group("resolver/amd/tests" FILES ${SOURCES_RESOLVER_AMD_TESTS})
source_group("resolver/cpu" FILES ${SOURCES_RESOLVER_CPU})
source_group("resolver/cpu/tests" FILES ${SOURCES_RESOLVER_CPU_TESTS})
source_group("resolver/nvidia" FILES ${SOURCES_RESOLVER_NVIDIA})
source_group("resolver/nvidia/tests" FILES ${SOURCES_RESOLVER_NVIDIA_TESTS})
source_group("stratums" FILES ${SOURCES_STRATUM})
source_group("statistical" FILES ${SOURCES_STRATISTICAL})

################################################################################
###  LINK LIBRARIES                                                         ####
################################################################################
if (NOT WIN32)
    target_link_options(${MINER_EXE} PUBLIC -static-libstdc++ -static-libgcc)
    target_link_options(${BENCH_EXE} PUBLIC -static-libstdc++ -static-libgcc)
    target_link_options(${UNIT_TEST_EXE} PUBLIC -static-libstdc++ -static-libgcc)
endif()

target_sources(${MINER_EXE} PUBLIC
    ${MINER_HEADERS}
    ${MINER_SOURCES}
)

target_sources(${BENCH_EXE} PUBLIC
    ${BENCH_HEADERS}
    ${BENCH_SOURCES}
)

target_sources(${UNIT_TEST_EXE} PUBLIC
    ${UNIT_TEST_HEADERS}
    ${UNIT_TEST_SOURCES}
)

include_directories(${MINER_EXE} PUBLIC
    ${Boost_INCLUDE_DIR}
    ${OPENSSL_INCLUDE_DIR}
    ${OpenCL_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

include_directories(${BENCH_EXE} PUBLIC
    ${Boost_INCLUDE_DIR}
    ${OpenCL_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

include_directories(${UNIT_TEST_EXE} PUBLIC
    ${Boost_INCLUDE_DIR}
    ${OPENSSL_INCLUDE_DIR}
    ${OpenCL_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${MINER_EXE}
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

target_link_libraries(${BENCH_EXE}
    ${Boost_LIBRARIES}
    ${OpenCL_LIBRARIES}
    ${CUDA_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIBRARY}
    ${CUDA_NVPTX_LIBRARY}
    ${CUDA_NVRTC_BUILTINS_LIBRARY}
    ${LIB_BENCH_NVIDIA}
)

target_link_libraries(${UNIT_TEST_EXE}
    gtest_main
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


if (UNIX)
    target_link_libraries(${MINER_EXE}
        ${GNUTLS_LIBRARIES}
        "stdc++fs"
    )
    target_link_libraries(${BENCH_EXE}
        ${GNUTLS_LIBRARIES}
        "stdc++fs"
    )
    target_link_libraries(${UNIT_TEST_EXE}
        ${GNUTLS_LIBRARIES}
        "stdc++fs"
    )
endif()
