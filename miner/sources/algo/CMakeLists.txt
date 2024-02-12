
set(ALGO_HEADERS
algo_type.hpp
bitwise.hpp
dag_context.hpp
hash.hpp
hash_utils.hpp
math.hpp
keccak.hpp
rol.hpp
)

set(ALGO_SOURCES
algo_type.cpp
hash_utils.cpp
keccak.cpp
math.cpp
)

target_sources(${MINER_EXE} PUBLIC
${ALGO_HEADERS}
${ALGO_SOURCES}
)

add_subdirectory(autolykos)
add_subdirectory(crypto)
add_subdirectory(ethash)
add_subdirectory(progpow)
