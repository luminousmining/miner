find_program(CLANG_FORMAT_EXECUTABLE
    NAMES clang-format-15 clang-format
)
if(CLANG_FORMAT_EXECUTABLE)
    add_custom_target(format
        COMMAND ${CMAKE_COMMAND}
            -DCLANG_FORMAT=${CLANG_FORMAT_EXECUTABLE}
            -DSOURCES_DIR=${CMAKE_CURRENT_SOURCE_DIR}/sources
            -DFIX=ON
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tools/clang_format.cmake
        COMMENT "Running clang-format on sources..."
        VERBATIM
    )
    add_custom_target(format-check
        COMMAND ${CMAKE_COMMAND}
            -DCLANG_FORMAT=${CLANG_FORMAT_EXECUTABLE}
            -DSOURCES_DIR=${CMAKE_CURRENT_SOURCE_DIR}/sources
            -DFIX=OFF
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tools/clang_format.cmake
        COMMENT "Checking clang-format on sources..."
        VERBATIM
    )
endif()
