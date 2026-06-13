if (TOOL_ANALYZER)
    find_program(CPPCHECK_EXECUTABLE cppcheck)
    if(CPPCHECK_EXECUTABLE)
        set(CMAKE_CXX_CPPCHECK
            ${CPPCHECK_EXECUTABLE}
            --enable=all
            --inconclusive
            --std=c++20
            --quiet
        )
    endif()

    find_program(CLANG_TIDY_EXECUTABLE
        NAMES clang-tidy-15 clang-tidy
    )
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
