################################################################################
###  LIB CUDA ETHASH                                                        ####
################################################################################
if (BUILD_NVIDIA)
    cuda_add_library(${LIB_CRYPTO_NVIDIA} STATIC
        sources/algo/autolykos/cuda/autolykos_v2.cu
        sources/algo/blake3/cuda/blake3.cu
        sources/algo/ethash/cuda/ethash.cu
        sources/algo/progpow/cuda/progpow.cu)

    target_compile_definitions(${LIB_CRYPTO_NVIDIA} PRIVATE __LIB_CUDA CUDA_ENABLE)

    include_directories(${LIB_CRYPTO_NVIDIA} PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/sources
    )
endif()

################################################################################
### HEADERS AND SOURCES FILES MINER                                         ####
################################################################################
set(MINER_HEADERS)
set(MINER_SOURCES
sources/miner.cpp
)

################################################################################
### HEADERS AND SOURCES FILES UNIT TEST                                     ####
################################################################################
set(UNIT_TEST_HEADERS)
set(UNIT_TEST_SOURCES
sources/unit_test.cpp
)

################################################################################
## LINK UNIT TEST                                                           ####
################################################################################
# Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(GoogleTest)

add_executable(${UNIT_TEST_EXE})
#gtest_discover_tests(${UNIT_TEST_EXE})

################################################################################
###  LINK MINER                                                             ####
################################################################################
add_executable(${MINER_EXE})

################################################################################
###  SOURCES                                                                ####
################################################################################
set(SOURCES_ALGO)
set(SOURCES_ALGO_AUTOLYOS)
set(SOURCES_ALGO_BLAKE3)
set(SOURCES_ALGO_CRYPTO)
set(SOURCES_ALGO_CRYPTO_TESTS)
set(SOURCES_ALGO_ETHASH)
set(SOURCES_ALGO_ETHASH_TESTS)
set(SOURCES_ALGO_PROGPOW)
set(SOURCES_ALGO_TESTS)
set(SOURCES_API)
set(SOURCES_COMMON)
set(SOURCES_COMMON_ERROR)
set(SOURCES_COMMON_LOG)
set(SOURCES_COMMON_MOCKER)
set(SOURCES_DEVICE)
set(SOURCES_NETWORK)
set(SOURCES_RESOLVER)
set(SOURCES_RESOLVER_AMD)
set(SOURCES_RESOLVER_AMD_TESTS)
set(SOURCES_RESOLVER_NVIDIA)
set(SOURCES_RESOLVER_NVIDIA_TESTS)
set(SOURCES_STRATISTICAL)
set(SOURCES_STRATUM)
add_subdirectory(sources)
source_group("" FILES sources/miner.cpp sources/unit_test.cpp)
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
source_group("common" FILES ${SOURCES_COMMON})
source_group("common/error" FILES ${SOURCES_COMMON_ERROR})
source_group("common/log" FILES ${SOURCES_COMMON_LOG})
source_group("common/mocker" FILES ${SOURCES_COMMON_MOCKER})
source_group("device" FILES ${SOURCES_DEVICE})
source_group("network" FILES ${SOURCES_NETWORK})
source_group("resolver" FILES ${SOURCES_RESOLVER})
source_group("resolver/amd" FILES ${SOURCES_RESOLVER_AMD})
source_group("resolver/amd/tests" FILES ${SOURCES_RESOLVER_AMD_TESTS})
source_group("resolver/nvidia" FILES ${SOURCES_RESOLVER_NVIDIA})
source_group("resolver/nvidia/tests" FILES ${SOURCES_RESOLVER_NVIDIA_TESTS})
source_group("stratums" FILES ${SOURCES_STRATUM})
source_group("statistical" FILES ${SOURCES_STRATISTICAL})

if (NOT WIN32)
    target_link_options(${MINER_EXE} PUBLIC -static-libstdc++ -static-libgcc)
    target_link_options(${UNIT_TEST_EXE} PUBLIC -static-libstdc++ -static-libgcc)
endif()

target_sources(${MINER_EXE} PUBLIC
    ${MINER_HEADERS}
    ${MINER_SOURCES}
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
    ${CMAKE_CURRENT_SOURCE_DIR}/sources
)

include_directories(${UNIT_TEST_EXE} PUBLIC
    ${Boost_INCLUDE_DIR}
    ${OPENSSL_INCLUDE_DIR}
    ${OpenCL_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/sources
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
    target_link_libraries(${UNIT_TEST_EXE}
        ${GNUTLS_LIBRARIES}
        "stdc++fs"
    )
endif()

