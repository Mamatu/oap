get_filename_component(a_dir "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
get_filename_component(a_dir "${a_dir}" NAME)
set(TARGET "${a_dir}")
set(INCLUDE_PATHS "oapMath")

list(APPEND DEPS oapMath)
list(APPEND DEPS oapMatrix)

set(EXTRA_LIBS )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
