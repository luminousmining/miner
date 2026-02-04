file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")


if (BUILD_EXE_UNIT_TEST)
    target_sources(${UNIT_TEST_EXE} PUBLIC
        ${HEADERS}
        ${SOURCES}
    )
endif()

set(SOURCES_ALGO_CRYPTO_TESTS ${HEADERS} ${SOURCES} PARENT_SCOPE)
