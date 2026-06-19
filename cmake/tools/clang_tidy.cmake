find_program(CLANG_TIDY_EXECUTABLE
    NAMES clang-tidy-15 clang-tidy
)

if (TOOL_ANALYZER)
    if(CLANG_TIDY_EXECUTABLE)
        set(CMAKE_CXX_CLANG_TIDY
            ${CLANG_TIDY_EXECUTABLE}
            --config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy
        )
        message(STATUS "clang-tidy enabled: ${CLANG_TIDY_EXECUTABLE}")
    else()
        message(WARNING "clang-tidy not found, skipping")
    endif()
endif()
