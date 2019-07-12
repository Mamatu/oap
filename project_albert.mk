MODE := Debug
PLATFORM := albert
USER_CXXOPTIONS :=
USER_NVCCOPTIONS := -arch=sm_20 --ptxas-options --verbose -maxrregcount=30
CXX := g++
NVCC := nvcc
