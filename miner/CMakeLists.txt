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

