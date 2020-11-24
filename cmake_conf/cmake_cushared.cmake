cmake_minimum_required(VERSION 3.5)

include (module.cmake)

project(oap CXX)

set_target_properties( PROPERTIES LINKER_LANGUAGE CXX) 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB source "*.h" "*.cpp")

add_library(${OAP_CMAKE_MODULE_NAME} SHARED source)

target_link_libraries(${OAP_CMAKE_MODULE_NAME} PUBLIC ${OAP_CMAKE_DEPS_LIBRARIES})

include_directories(ArnoldiPackage oap2dt3dDevice oapAppUtils oapAppUtilsTests oapCMatrixDataTests oapCuda oapDeviceTestsData oapHostTests oapMath oapMatrix oapMatrixCpu oapMatrixCuda oapMemory oapNeural oapNeuralApps oapNeuralDevice oapNeuralHost oapNeuralRoutines oapNeuralRoutinesHost oapQRTestSamples oapTests oapTestsData oapUtils)
