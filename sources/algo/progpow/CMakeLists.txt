file(GLOB HEADERS "*.hpp")
file(GLOB SOURCES "*.cpp")

if (BUILD_EXE_MINER)
  target_sources(${MINER_EXE} PUBLIC
    ${HEADERS}
    ${SOURCES}
  )
endif()

if (BUILD_EXE_UNIT_TEST)
  target_sources(${UNIT_TEST_EXE} PUBLIC
    ${HEADERS}
    ${SOURCES}
  )
endif()

set(OUT_PROGPOW ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/progpow)

file(MAKE_DIRECTORY ${OUT_PROGPOW})

set(PROGPOW_FILES
  result.hpp
)

foreach(FILE ${PROGPOW_FILES})
  add_custom_command(
    OUTPUT ${OUT_PROGPOW}/${FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${OUT_PROGPOW}/${FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    COMMENT "Copy ${FILE}"
  )
endforeach()

add_custom_target(copy_progpow_cpp_files ALL
  DEPENDS
    ${OUT_PROGPOW}/result.hpp
)

add_subdirectory(cuda)
add_subdirectory(opencl)

set(SOURCES_ALGO_PROGPOW ${HEADERS} ${SOURCES} PARENT_SCOPE)
