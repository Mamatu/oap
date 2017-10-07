MODE := Debug
PLATFORM := albert
EXTRA_CXXOPTIONS :=
EXTRA_NVCCOPTIONS := -arch=sm_20 --ptxas-options --verbose -maxrregcount=30
CXX := g++
NVCC := nvcc
