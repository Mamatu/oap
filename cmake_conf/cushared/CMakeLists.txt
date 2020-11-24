cmake_minimum_required(VERSION 3.17)

include (module.cmake)

file(GLOB source "*.h" "*.cpp" "*.c")
file(GLOB cu_source "*.h" "*.cu")

add_library(${TARGET} SHARED ${source})
add_library(${TARGET}_cu OBJECT ${cu_source})

set_property(TARGET ${TARGET}_cu PROPERTY CUDA_PTX_COMPILATION ON)

add_definitions(-DOAP_CUDA_BUILD)

if(DEPS)
add_dependencies(${TARGET} ${DEPS})
endif()

target_link_libraries(${TARGET} PUBLIC -lcuda ${DEPS} ${EXTRA_LIBS})
target_include_directories(${TARGET} PUBLIC ${INCLUDE_PATH})
