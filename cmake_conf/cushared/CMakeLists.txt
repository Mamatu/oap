if("${BUILD_PART}" STREQUAL "ALL")

include (module.cmake)

file(GLOB source "*.h" "*.cpp" "*.c")

add_library(${TARGET} SHARED ${source})

add_definitions(-DOAP_CUDA_BUILD)

if(DEPS)
  add_dependencies(${TARGET} ${DEPS} ${CU_TARGET})
endif()

target_link_libraries(${TARGET} PUBLIC -lcuda ${DEPS} ${EXTRA_LIBS})
target_include_directories(${TARGET} PUBLIC ${INCLUDE_PATH})

endif()
