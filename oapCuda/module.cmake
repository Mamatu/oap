get_filename_component(a_dir "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
get_filename_component(a_dir "${a_dir}" NAME)
set(TARGET "${a_dir}")
set(INCLUDE_PATHS "oapMath -I/usr/local/cuda/include")
set(EXTRA_LIBS -lcuda)
