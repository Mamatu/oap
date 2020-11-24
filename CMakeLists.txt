cmake_minimum_required(VERSION 3.19)

project(oap CXX CUDA)

set_target_properties( PROPERTIES LINKER_LANGUAGE CXX) 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB gmock_source $ENV{OAP_GMOCK_PATH}/src/gmock-all.cc)
file(GLOB gtest_source $ENV{OAP_GTEST_PATH}/src/gtest-all.cc $ENV{OAP_GTEST_PATH}/src/gtest_main.cc)

add_library(gmock SHARED ${gmock_source})
add_library(gtest SHARED ${gtest_source})

target_include_directories(gmock PUBLIC $ENV{OAP_GMOCK_PATH}/include $ENV{OAP_GMOCK_PATH} $ENV{OAP_GTEST_PATH}/include)
target_include_directories(gtest PUBLIC $ENV{OAP_GTEST_PATH}/include $ENV{OAP_GTEST_PATH})

target_link_libraries(gmock PUBLIC -lpthread)
target_link_libraries(gtest PUBLIC -lpthread)


#shared libraries
list(APPEND LIBS oapMath)
list(APPEND LIBS oapMatrix)
list(APPEND LIBS oapUtils)
list(APPEND LIBS oapAppUtils)
list(APPEND LIBS oapCuda)
list(APPEND LIBS oapMatrixCpu)
list(APPEND LIBS oapMemory)
list(APPEND LIBS oapNeuralHost)
list(APPEND LIBS oapNeuralDevice)
list(APPEND LIBS oapNeuralRoutines)
list(APPEND LIBS oapNeuralRoutinesHost)
list(APPEND LIBS ArnoldiPackage)

#cushared libraries
list(APPEND LIBS oapMatrixCuda)
list(APPEND LIBS oap2dt3dDevice)

#executables
list(APPEND EXECS oap2dt3d)
list(APPEND EXECS oapNeuralApps)
list(APPEND EXECS oapNeural)

#executables tests
list(APPEND EXECS oapAppUtilsFuncTests)
list(APPEND EXECS oapAppUtilsTests)
list(APPEND EXECS oapArnoldiHostTests)
#list(APPEND EXECS oapCMatrixDataTests)
list(APPEND EXECS oapHostTests)
list(APPEND EXECS oapNeuralRoutinesHostTests)

#executables cu tests
list(APPEND EXECS oap2dt3dDeviceTests)
list(APPEND EXECS oapArnoldiDeviceTests)
list(APPEND EXECS oapDeviceTests)
list(APPEND EXECS oapNeuralDeviceTests)

foreach(lib IN LISTS LIBS)
  include_directories (${lib})
  link_directories (build/${lib})
endforeach()

foreach(lib IN LISTS EXECS)
  include_directories (${lib})
endforeach()

include_directories (oapTests)
include_directories (oapTestsData)
include_directories (oapQRTestSamples)
include_directories (oapCMatrixDataTests)
include_directories (oapDeviceTestsData)

include_directories ($ENV{OAP_CUDA_PATH}/include)

add_definitions (-DDEBUG -DOAP_PATH=/home/mmatula/oap -DTMP_PATH=/tmp/Oap -DOAP_CONFIG_NI_EF -DLOG_LEVEL=DEBUG_LEVEL)

foreach(lib IN LISTS LIBS)
  add_subdirectory (${lib})
endforeach()
foreach(exec IN LISTS EXECS)
  add_subdirectory (${exec})
endforeach()
