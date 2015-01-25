include ../project_generic.mk
include module.mk
OGLA_INCLUDES_PATHS := $(addprefix -I/$(OGLA_PATH)/, $(OGLA_INCLUDES))
CU_FILES := $(wildcard *.cu)
CPP_FILES := $(wildcard *.cpp)	
OCPP_FILES := $(addprefix build/$(MODE)/$(PLATFORM)/,$(notdir $(CPP_FILES:.cpp=.o)))
OCU_FILES := $(addprefix build/$(MODE)/$(PLATFORM)/,$(notdir $(CU_FILES:.cu=.o)))
CXX := g++
NVCC := $(CUDA_PATH)/bin/nvcc --compiler-bindir /usr/bin/gcc-4.8
INCLUDE_DIRS := -I/usr/local/cuda/include -I/usr/include $(OGLA_INCLUDES_PATHS)
INCLUDE_DIRS += $(INCLUDE_PATHS)
NVCC_INCLUDE_DIRS := -I/usr/local/cuda/include
LIBS_DIRS := -L/usr/lib/nvidia-current -L/usr/lib -L/usr/local/cuda/lib64
LIBS := $(EXTRA_LIBS)

ifeq ($(MODE), Debug)
	CXXOPTIONS := -c -g -fPIC -D$(TYPES) -D$(KERNEL_INFO)
	CXXOPTIONS += $(EXTRA_CXXOPTIONS)
	NVCCOPTIONS := -g -G -DEXTENDED_TYPES
	NVCCOPTIONS += $(EXTRA_NVCCOPTIONS)
endif

ifeq ($(MODE), Release)
	CXXOPTIONS := -c -O2 -fPIC -D$(TYPES) -D$(KERNEL_INFO)
	CXXOPTIONS += $(EXTRA_CXXOPTIONS)
	NVCCOPTIONS := -O2 -DEXTENDED_TYPES
	NVCCOPTIONS += $(EXTRA_NVCCOPTIONS)
endif
