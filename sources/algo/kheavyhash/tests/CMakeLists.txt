file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")

if (BUILD_EXE_UNIT_TEST)
    target_sources(${UNIT_TEST_EXE} PUBLIC
        ${HEADERS}
        ${SOURCES}
    )
    target_include_directories(${UNIT_TEST_EXE} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

set(SOURCES_ALGO_KHEAVYHASH_TESTS ${HEADERS} ${SOURCES} PARENT_SCOPE)
