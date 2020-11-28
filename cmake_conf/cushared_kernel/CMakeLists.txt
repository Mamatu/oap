if("${BUILD_PART}" STREQUAL "ALL")

cmake_minimum_required(VERSION 3.19)
include (module.cmake)

file(GLOB_RECURSE source "*.cpp" "*.c")
file(GLOB_RECURSE cu_source "*.cu")

add_library(${TARGET} SHARED ${source})

add_definitions(-DOAP_CUDA_BUILD)

set (CUDA_INCLUDE_DIRS "")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  list (APPEND CUDA_INCLUDE_DIRS_LIST "-I${dir}")
endforeach()
set (CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS_LIST})
message("CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}")

set (CUDA_DEFINITIONS "")
get_property(defs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY COMPILE_DEFINITIONS)
foreach(def ${defs})
  list (APPEND CUDA_DEFINITIONS_LIST "-D${def} ")
endforeach()
set (CUDA_DEFINITIONS ${CUDA_DEFINITIONS_LIST})
message("CUDA_DEFINITIONS ${CUDA_DEFINITIONS}")
message("NVCC_COMPILATION_OPTIONS ${NVCC_COMPILATION_OPTIONS}")

set (CUBIN_TARGET "lib${TARGET}.cubin")

add_custom_command(OUTPUT ${CUBIN_TARGET} COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_COMPILATION_OPTIONS} ${CUDA_DEFINITIONS} ${CUDA_INCLUDE_DIRS} --cubin ${cu_source} -o ${CUBIN_TARGET} DEPENDS ${cu_source})
add_custom_command(OUTPUT copy_cubin POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy *.cubin ${CMAKE_BINARY_DIR}/cubin/ DEPENDS ${CUBIN_TARGET})

add_custom_target(CU_${TARGET} ALL DEPENDS ${CUBIN_TARGET} copy_cubin)

if(DEPS)
  add_dependencies(${TARGET} ${DEPS} ${CU_TARGET})
endif()

target_link_libraries(${TARGET} PUBLIC -lcuda ${DEPS} ${EXTRA_LIBS})
target_include_directories(${TARGET} PUBLIC ${INCLUDE_PATH})

#file(COPY "*.cubin" DESTINATION "cubin")

endif()
