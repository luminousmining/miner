set(RESOLVER_HEADERS
resolver.hpp
)

set(RESOLVER_SOURCES
resolver.cpp)

target_sources(${MINER_EXE} PUBLIC
${RESOLVER_HEADERS}
${RESOLVER_SOURCES}
)

add_subdirectory(amd)
add_subdirectory(nvidia)
