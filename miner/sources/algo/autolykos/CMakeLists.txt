set(AUTOLYKOS_HEADERS
autolykos.hpp
result.hpp
)

set(AUTOLYKOS_SOURCES
autolykos.cpp
)

target_sources(${MINER_EXE} PUBLIC
${AUTOLYKOS_HEADERS}
${AUTOLYKOS_SOURCES}
)

add_subdirectory(opencl)
