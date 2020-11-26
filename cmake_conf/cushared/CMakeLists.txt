if("${BUILD_PART}" STREQUAL "ALL")

cmake_minimum_required(VERSION 3.19)
include (module.cmake)

file(GLOB source "*.h" "*.cpp" "*.c")
file(GLOB cu_source "*.h" "*.cu")
file(GLOB cu_ptx "*.ptx")

#set(CU_TARGET cu${TARGET})

add_library(${TARGET} SHARED ${source})
#add_library(${CU_TARGET} OBJECT ${cu_source})

#set_property(TARGET ${CU_TARGET} PROPERTY CUDA_PTX_COMPILATION ON)
#set_property(TARGET ${CU_TARGET} PROPERTY CUDA_ARCHITECTURES 30)

#$(NVCC) -DOAP_CUDA_BUILD $(NVCCOPTIONS) $(NVCC_INCLUDE_DIRS) $(INCLUDE_DIRS) --cubin $(CU_FILES) -o $@.cubin

add_custom_command(OUTPUT ${TARGET}.cubin_1 POST_BUILD COMMAND ${CMAKE_COMMAND} -E aa${NVCC} --cubin ${cu_source} -o ${TARGET}.cubin DEPENDS ${cu_source})
add_custom_target(${TARGET}.cubin DEPENDS ${TARGET}.cubin_1)

add_definitions(-DOAP_CUDA_BUILD)

if(DEPS)
  add_dependencies(${TARGET} ${DEPS} ${CU_TARGET})
endif()

target_link_libraries(${TARGET} PUBLIC -lcuda ${DEPS} ${EXTRA_LIBS})
target_include_directories(${TARGET} PUBLIC ${INCLUDE_PATH})

endif()
