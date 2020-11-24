cmake_minimum_required(VERSION 3.19)

include (module.cmake)

file(GLOB source "*.h" "*.cpp" "*.c")

add_executable(${TARGET} ${source})

add_definitions(-DOAP_CUDA_BUILD)

target_link_libraries(${TARGET} PUBLIC gmock gtest ${EXTRA_LIBS})
target_include_directories(${TARGET} PUBLIC ${INCLUDE_PATH})
