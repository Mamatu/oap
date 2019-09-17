MODE := Debug
PLATFORM := x86
USER_CXXOPTIONS :=
USER_NVCCOPTIONS := -arch=sm_30 --ptxas-options --verbose
CXX := g++-5
NVCC := nvcc --compiler-bindir /usr/bin/gcc
