if("${BUILD_PART}" STREQUAL "ALL")

include (module.cmake)

file(GLOB source "*.h" "*.cpp" "*.c")


if("${LINKING}" STREQUAL "DYNAMIC")
  add_library(${TARGET} SHARED ${source})
elseif("${LINKING}" STREQUAL "STATIC")
  add_library(${TARGET} STATIC ${source})
endif()

add_definitions(-DOAP_CUDA_BUILD)

if(DEPS)
  add_dependencies(${TARGET} ${DEPS} ${CU_TARGET})
endif()

target_link_libraries(${TARGET} PUBLIC -lcuda ${DEPS} ${EXTRA_LIBS})
target_include_directories(${TARGET} PUBLIC ${INCLUDE_PATH})

endif()
