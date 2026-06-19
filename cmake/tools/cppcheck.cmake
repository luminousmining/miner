find_program(CPPCHECK_EXECUTABLE cppcheck)

cmake_language(DEFER CALL cmake_language EVAL CODE [[
    if(CPPCHECK_EXECUTABLE AND BUILD_EXE_MINER)
        get_target_property(_miner_sources ${MINER_EXE} SOURCES)
        list(FILTER _miner_sources INCLUDE REGEX ".*\\.cpp$")

        get_target_property(_miner_includes ${MINER_EXE} INCLUDE_DIRECTORIES)
        set(_cppcheck_includes)
        if(_miner_includes)
            foreach(_dir ${_miner_includes})
                if(NOT _dir MATCHES "^\\$<")
                    list(APPEND _cppcheck_includes "-I${_dir}")
                endif()
            endforeach()
        endif()

        add_custom_target(cppcheck
            COMMAND ${CPPCHECK_EXECUTABLE}
                --enable=all
                --inconclusive
                --std=c++20
                --quiet
                ${_cppcheck_includes}
                ${_miner_sources}
            COMMENT "Running cppcheck on MINER_EXE sources..."
            VERBATIM
        )
    endif()
]])
