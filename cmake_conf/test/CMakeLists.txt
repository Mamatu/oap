
include (module.cmake)

file(GLOB source "*.h" "*.cpp" "*.c")

add_executable(${TARGET} ${source})

if(DEPS)
add_dependencies(${TARGET} ${DEPS})
endif()

target_link_libraries(${TARGET} PUBLIC gmock gtest ${DEPS} ${EXTRA_LIBS})
target_include_directories(${TARGET} PUBLIC ${INCLUDE_PATH})
