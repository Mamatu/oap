include ../project_generic.mk
include module.mk
OGLA_INCLUDES_PATHS := $(addprefix -I/$(OGLA_PATH)/, $(OGLA_INCLUDES))
CU_FILES := $(wildcard *.cu)
CPP_FILES := $(wildcard *.cpp)	
OCPP_FILES := $(addprefix build/$(MODE)/$(PLATFORM)/,$(notdir $(CPP_FILES:.cpp=.o)))
OCU_FILES := $(addprefix build/$(MODE)/$(PLATFORM)/,$(notdir $(CU_FILES:.cu=.o)))
CXX := g++
NVCC := $(CUDA_PATH)/bin/nvcc
INCLUDE_DIRS := -I/usr/local/cuda/include -I/usr/local/cuda-6.0/include  -I/usr/include -I/usr/X11R6/include $(OGLA_INCLUDES_PATHS)
INCLUDE_DIRS += $(INCLUDE_PATHS)
NVCC_INCLUDE_DIRS := -I/usr/local/cuda/include -I/usr/local/cuda-6.0/include
LIBS_DIRS := -L/usr/lib/nvidia-current -L/usr/lib -L/usr/local/lib -L/usr/local/cuda-6.0/lib64 -L/usr/lib/i386-linux-gnu
LIBS := $(EXTRA_LIBS)

ifeq ($(MODE), Debug)
	CXXOPTIONS := -fPIC -c -g
	NVCCOPTIONS := -g #-arch=sm_30
endif

ifeq ($(MODE), Release)
	CXXOPTIONS := -c -fPIC -O2
	NVCCOPTIONS := -O #-arch=sm_30
endif
