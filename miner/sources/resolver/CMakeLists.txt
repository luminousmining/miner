set(RESOLVER_HEADERS
resolver.hpp
)

set(RESOLVER_SOURCES
resolver.cpp)

target_sources(${MINER_EXE} PUBLIC
${RESOLVER_HEADERS}
${RESOLVER_SOURCES}
)

target_sources(${UNIT_TEST_EXE} PUBLIC
${RESOLVER_HEADERS}
${RESOLVER_SOURCES}
)

if (BUILD_AMD)
    add_subdirectory(amd)
endif()
if (BUILD_NVIDIA)
    add_subdirectory(nvidia)
endif()
