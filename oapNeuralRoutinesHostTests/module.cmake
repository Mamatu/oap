get_filename_component(a_dir "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
get_filename_component(a_dir "${a_dir}" NAME)
set(TARGET "${a_dir}")
set(INCLUDE_PATHS "")

list(APPEND DEPS oapAppUtils)
list(APPEND DEPS oapMatrixCpu)
list(APPEND DEPS oapMatrix)
list(APPEND DEPS oapNeuralRoutinesHost)
list(APPEND DEPS oapUtils)

set(EXTRA_LIBS )
