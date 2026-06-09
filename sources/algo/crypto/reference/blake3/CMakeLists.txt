# Vendored official BLAKE3 C reference, compiled as a C static library so both the
# FishHash reference (algo/fishhash) and the Alephium PoW test (algo/blake3) can link a
# single canonical implementation. blake3_neon.c is ARM-only and excluded on x86.
project(blake3_ref C)

add_library(blake3_ref STATIC
    blake3.c
    blake3_portable.c
    blake3_dispatch.c
)

target_include_directories(blake3_ref PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
