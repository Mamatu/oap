MODE := Debug
PLATFORM := albert
USER_CXXOPTIONS :=
USER_NVCCOPTIONS := -arch=sm_30 --ptxas-options --verbose
CXX := g++-4.4
NVCC := nvcc --compiler-bindir /usr/bin/gcc-4.8
