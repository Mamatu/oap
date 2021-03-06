cmake_minimum_required(VERSION 3.17)

set_target_properties( PROPERTIES LINKER_LANGUAGE CXX) 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(OAP_PATH "${CMAKE_SOURCE_DIR}" CACHE STRING "Path to project.")
set(TMP_PATH "${CMAKE_CURRENT_BINARY_DIR}/Oap" CACHE STRING "Path to temporary directory for data etc.")
set (BUILD_MODE "DEBUG" CACHE STRING "Build mode. Can be: DEBUG, RELEASE, DEBUG_SANITIZER")
set (BUILD_PART "ALL" CACHE STRING "Build specific part of s/w. In the case of ALL device and host parts will be compiled. Can be: ALL, HOST")
set (LOG_LEVEL "DEBUG_LEVEL" CACHE STRING "Log level")
set (MATH_TYPE_CONFIGURATION "NI_EF" CACHE STRING "Type of math values NI_EF, EI_EF")
set (CPU_PROFILING "N" CACHE STRING "Add pg options for compiling (profiling). It will turn on also LINKING STATIC mode")
set (LINKING "DYNAMIC" CACHE STRING "Choose linking options: DYNAMIC or STATIC")

set(NVCC_COMPILATION_OPTIONS "")

add_definitions (-DLOG_LEVEL=${LOG_LEVEL} -DOAP_PATH=${OAP_PATH} -DTMP_PATH=${TMP_PATH})

if("${BUILD_PART}" STREQUAL "ALL")
  project(oap CXX CUDA)
elseif("${BUILD_PART}" STREQUAL "HOST")
  project(oap CXX)
endif()

file(MAKE_DIRECTORY ${TMP_PATH}/tests_data)
file(MAKE_DIRECTORY ${TMP_PATH}/conversion_data)
file(MAKE_DIRECTORY ${TMP_PATH}/host_tests)
file(MAKE_DIRECTORY ${TMP_PATH}/host_tests/OapPngFileTests_PrintLetters)
file(MAKE_DIRECTORY ${TMP_PATH}/device_tests)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cubin)

if("${MATH_TYPE_CONFIGURATION}" STREQUAL "NI_EF")
  add_definitions(-DOAP_CONFIG_NI_EF)
elseif("${MATH_TYPE_CONFIGURATION}" STREQUAL "EI_EF")
  add_definitions(-DOAP_CONFIG_EI_EF)
else()
  message(FATAL_ERROR "This kind of MATH_TYPE_CONFIGURATION is not supported")
endif()

if("${BUILD_MODE}" STREQUAL "DEBUG")
  add_definitions(-DDEBUG)
  set(CMAKE_CXX_FLAGS "-g3")
  set(NVCC_COMPILATION_OPTIONS "-g")
elseif("${BUILD_MODE}" STREQUAL "RELEASE")
  set(CMAKE_CXX_FLAGS "-O2")
  set(NVCC_COMPILATION_OPTIONS "-O2")
elseif("${BUILD_MODE}" STREQUAL "DEBUG_SANITIZER")
  add_definitions(-DDEBUG)
  add_definitions(-DOAP_DISABLE_ABORT_MEMLEAK)
  add_definitions(-DOAP_DISABLE_ASSERTION)
  set(CMAKE_CXX_FLAGS "-g3 -fsanitize=address -fno-omit-frame-pointer")
  set(NVCC_COMPILATION_OPTIONS "-g")
else()
  message(FATAL_ERROR "This kind of BUILD_MODE is not supported")
endif()


if("${CPU_PROFILING}" STREQUAL "Y")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set (LINKING "STATIC")
endif()

if("${LINKING}" STREQUAL "STATIC")
  
endif()

if(NOT("${BUILD_PART}" STREQUAL "ALL" OR "${BUILD_PART}" STREQUAL "HOST"))
  message(FATAL_ERROR "This kind of BUILD_PART is not supported")
endif()

file(GLOB gmock_source $ENV{OAP_GMOCK_PATH}/src/gmock-all.cc)
file(GLOB gtest_source $ENV{OAP_GTEST_PATH}/src/gtest-all.cc $ENV{OAP_GTEST_PATH}/src/gtest_main.cc)

add_library(gmock SHARED ${gmock_source})
add_library(gtest SHARED ${gtest_source})

target_include_directories(gmock PUBLIC $ENV{OAP_GMOCK_PATH}/include $ENV{OAP_GMOCK_PATH} $ENV{OAP_GTEST_PATH}/include)
target_include_directories(gtest PUBLIC $ENV{OAP_GTEST_PATH}/include $ENV{OAP_GTEST_PATH})

target_link_libraries(gmock PUBLIC -lpthread)
target_link_libraries(gtest PUBLIC -lpthread)

list(APPEND LIBS oapMath)
list(APPEND LIBS oapMatrix)
list(APPEND LIBS oapUtils)
list(APPEND LIBS oapAppUtils)
list(APPEND LIBS oapMatrixCpu)
list(APPEND LIBS oapMemory)
list(APPEND LIBS oapNeuralGeneric)
list(APPEND LIBS oapNeuralRoutines)
list(APPEND LIBS oapNeuralRoutinesHost)
list(APPEND LIBS ArnoldiPackage)
list(APPEND LIBS oapMatrixCuda)
list(APPEND LIBS oap2dt3dDevice)
list(APPEND LIBS oapNeuralDevice)
list(APPEND LIBS oapNeuralHost)

list(APPEND EXECS oap2dt3d)
list(APPEND EXECS oapNeuralApps)
list(APPEND EXECS oapNeural)
list(APPEND EXECS oapAppUtilsFuncTests)
list(APPEND EXECS oapAppUtilsTests)
list(APPEND EXECS oapArnoldiHostTests)
list(APPEND EXECS oapHostTests)
list(APPEND EXECS oapNeuralRoutinesHostTests)
list(APPEND EXECS oap2dt3dDeviceTests)
list(APPEND EXECS oapArnoldiDeviceTests)
list(APPEND EXECS oapDeviceTests)
list(APPEND EXECS oapNeuralDeviceTests)
list(APPEND EXECS oapNeuralHostTests)

foreach(lib IN LISTS LIBS)
  include_directories (${lib})
  link_directories (build/${lib})
endforeach()

foreach(lib IN LISTS EXECS)
  include_directories (${lib})
endforeach()

include_directories (oapTests)
include_directories (oapCuda)
include_directories (oapTestsData)
include_directories (oapQRTestSamples)
include_directories (oapCMatrixDataTests)
include_directories (oapDeviceTestsData)

if("${BUILD_PART}" STREQUAL "ALL")
  include_directories ($ENV{OAP_CUDA_PATH}/include)
endif()

foreach(lib IN LISTS LIBS)
  add_subdirectory (${lib})
endforeach()
foreach(exec IN LISTS EXECS)
  add_subdirectory (${exec})
endforeach()
