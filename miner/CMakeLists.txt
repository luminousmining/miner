################################################################################
###  LIB CUDA ETHASH                                                        ####
################################################################################
if (BUILD_NVIDIA)
    cuda_add_library(${LIB_CRYPTO_NVIDIA} STATIC
        sources/algo/autolykos/cuda/autolykos_v2.cu
        sources/algo/blake3/cuda/blake3.cu
        sources/algo/ethash/cuda/ethash.cu
        sources/algo/progpow/cuda/progpow.cu)

    target_compile_definitions(${LIB_CRYPTO_NVIDIA} PRIVATE __LIB_CUDA)

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
add_subdirectory(sources)

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

