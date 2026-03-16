# clang_format.cmake
# Called by the 'format' and 'format-check' CMake targets.
#
# Parameters:
#   CLANG_FORMAT  - path to clang-format executable
#   SOURCES_DIR   - root directory to scan
#   FIX           - ON  -> format in-place (-i)
#                   OFF -> dry-run (error if file is not formatted)

if(NOT CLANG_FORMAT)
    message(FATAL_ERROR "CLANG_FORMAT not set")
endif()
if(NOT SOURCES_DIR)
    message(FATAL_ERROR "SOURCES_DIR not set")
endif()

file(GLOB_RECURSE ALL_FILES
    "${SOURCES_DIR}/*.cpp"
    "${SOURCES_DIR}/*.hpp"
)

set(ERRORS 0)

foreach(FILE ${ALL_FILES})
    # Skip CUDA and OpenCL kernel files
    if(FILE MATCHES ".*/cuda/.*" OR FILE MATCHES ".*/opencl/.*")
        continue()
    endif()

    if(FIX)
        execute_process(
            COMMAND ${CLANG_FORMAT} --style=file -i "${FILE}"
            RESULT_VARIABLE RESULT
            ERROR_VARIABLE  ERR_OUTPUT
        )
        if(NOT RESULT EQUAL 0)
            message(WARNING "clang-format failed on: ${FILE}\n${ERR_OUTPUT}")
            math(EXPR ERRORS "${ERRORS} + 1")
        endif()
    else()
        execute_process(
            COMMAND ${CLANG_FORMAT} --style=file --dry-run --Werror "${FILE}"
            RESULT_VARIABLE RESULT
            ERROR_VARIABLE  ERR_OUTPUT
        )
        if(NOT RESULT EQUAL 0)
            message(WARNING "Format error: ${FILE}\n${ERR_OUTPUT}")
            math(EXPR ERRORS "${ERRORS} + 1")
        endif()
    endif()
endforeach()

if(ERRORS GREATER 0)
    if(FIX)
        message(FATAL_ERROR "clang-format failed on ${ERRORS} file(s).")
    else()
        message(FATAL_ERROR "${ERRORS} file(s) are not correctly formatted. Run: cmake --build . --target format")
    endif()
else()
    if(FIX)
        message(STATUS "clang-format: all files formatted.")
    else()
        message(STATUS "clang-format: all files are correctly formatted.")
    endif()
endif()
