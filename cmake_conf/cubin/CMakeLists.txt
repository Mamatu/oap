if("${BUILD_PART}" STREQUAL "ALL")

cmake_minimum_required(VERSION 3.17)
include (module.cmake)

file(GLOB source "*.h" "*.cpp" "*.c")

add_executable(${TARGET} ${source})

add_definitions(-DOAP_CUDA_BUILD)

if(DEPS)
add_dependencies(${TARGET} ${DEPS})
endif()

target_link_libraries(${TARGET} PUBLIC ${DEPS} ${EXTRA_LIBS})
target_include_directories(${TARGET} PUBLIC ${INCLUDE_PATH})

endif()
