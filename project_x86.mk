MODE := Debug
PLATFORM := x86
EXTRA_CXXOPTIONS :=
EXTRA_NVCCOPTIONS := -arch=sm_30 --ptxas-options --verbose 
CXX := g++
NVCC := nvcc --compiler-bindir /usr/bin/gcc
