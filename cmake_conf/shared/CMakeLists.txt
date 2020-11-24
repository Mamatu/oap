cmake_minimum_required(VERSION 3.19)

include (module.cmake)

project(oap CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB source "*.h" "*.cpp" "*.c")

add_library(${TARGET} SHARED ${source})

target_link_libraries(${TARGET} PUBLIC ${EXTRA_LIBS})
target_include_directories(${TARGET} PUBLIC ${INCLUDE_PATH})
