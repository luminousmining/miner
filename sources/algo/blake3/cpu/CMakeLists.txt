# Vendored official BLAKE3 C reference, compiled as a C static library. It backs the host
# PoW oracle (algo/blake3/blake3_pow.cpp) and the shared crypto BLAKE3 KAT
# (algo/crypto/opencl/tests); any consumer can link the canonical blake3_ref target.
# blake3_neon.c is ARM-only and excluded on x86.
project(blake3_ref CXX)

set_source_files_properties(
    blake3.c
    blake3_portable.c
    blake3_dispatch.c
    PROPERTIES LANGUAGE CXX
)

add_library(blake3_ref STATIC
    blake3.c
    blake3_portable.c
    blake3_dispatch.c
)

target_include_directories(blake3_ref PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Future CPU resolver (resolver/cpu/blake3.cpp): BLAKE3 has no DAG -- loop nonces,
# call algo::blake3::hashRef + chainIndex, fill algo::blake3::Result (all device-
# neutral, already shared in algo/blake3). The winner predicate (digest <= target +
# group match) and the 24-byte submit-nonce hex currently live in resolver/amd/blake3.cpp;
# lift them to algo/blake3 when that second consumer exists. To actually mine (not just
# test), compile blake3_pow.cpp + link blake3_ref into MINER_EXE under BUILD_CPU --
# today both are unit_test-only (see algo/blake3/CMakeLists.txt).
