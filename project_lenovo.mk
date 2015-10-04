CUDA_PATH := /usr/local/cuda
MODE := Debug
PLATFORM := lenovo
EXTRA_CXXOPTIONS :=
EXTRA_NVCCOPTIONS := -arch=sm_30 --ptxas-options --verbose 
CXX := g++
NVCC := $(CUDA_PATH)/bin/nvcc --compiler-bindir /usr/bin/gcc-4.8
