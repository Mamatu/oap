CUDA_PATH := /usr/local/cuda
MODE := Debug
PLATFORM := albert
EXTRA_CXXOPTIONS :=
EXTRA_NVCCOPTIONS := -arch=sm_20 --ptxas-options --verbose -maxrregcount=30
CXX := g++
NVCC := $(CUDA_PATH)/bin/nvcc
