set(CRYPTO_HEADERS
fnv1.hpp
kiss99.hpp
)

set(CRYPTO_SOURCES
fnv1.cpp
kiss99.cpp
)

target_sources(${MINER_EXE} PUBLIC
${CRYPTO_HEADERS}
${CRYPTO_SOURCES}
)
