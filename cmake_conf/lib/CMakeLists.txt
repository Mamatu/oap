include (module.cmake)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB source "*.h" "*.cpp" "*.c")

if("${LINKING}" STREQUAL "DYNAMIC")
  add_library(${TARGET} SHARED ${source})
elseif("${LINKING}" STREQUAL "STATIC")
  add_library(${TARGET} STATIC ${source})
endif()

if(DEPS)
add_dependencies(${TARGET} ${DEPS})
endif()

target_link_libraries(${TARGET} PUBLIC ${DEPS} ${EXTRA_LIBS})
target_include_directories(${TARGET} PUBLIC ${INCLUDE_PATH})
