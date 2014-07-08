CPP_FILES = $(wildcard *.cpp)
CU_FILES = $(wildcard *.cu)
MODE := Release
PLATFORM := GNU-Linux-x86
LD_CUDA_PATH:=/usr/local/cuda-6.0
OCPP_FILES := $(addprefix build/$(MODE)/$(PLATFORM)/,$(notdir $(CPP_FILES:.cpp=.o)))
OCU_FILES := $(addprefix build/$(MODE)/$(PLATFORM)/,$(notdir $(CU_FILES:.cu=.o)))
OGLA_PATH = /home/mmatula/Ogla
TARGET := liboglaShibataCuda
CXX := g++
NVCC := $(LD_CUDA_PATH)/bin/nvcc
OGLA_PROJECTS_INCLUDE_DIRS = -I$(OGLA_PATH)/oglaServerUtils -I$(OGLA_PATH)/oglaUtils -I$(OGLA_PATH)/oglaMath  -I$(OGLA_PATH)/oglaMatrix -I$(OGLA_PATH)/oglaMatrixCuda -I$(OGLA_PATH)/oglaMatrixCpu -I$(OGLA_PATH)/oglaShibataCpu
INCLUDE_DIRS := -I/usr/local/cuda/include -I/usr/local/cuda-6.0/include  -I/usr/include -I/usr/X11R6/include
INCLUDE_DIRS += $(OGLA_PROJECTS_INCLUDE_DIRS)
NVCC_INCLUDE_DIRS := -I/usr/local/cuda/include -I/usr/local/cuda-6.0/include -I$(OGLA_PATH)/oglaMath -I$(OGLA_PATH)/oglaMatrix -I$(OGLA_PATH)/oglaShibataCpu
LIBS_DIRS := -L/usr/lib/nvidia-current -L/usr/lib -L/usr/local/lib -L/usr/local/cuda-6.0/lib64 -L/usr/lib/i386-linux-gnu -L/usr/lib/nvidia-331 -L/usr/lib/nvidia-331-updates
LIBS := -lcudart -lcuda

ifeq ($(MODE), Debug)
	CXXOPTIONS := -fPIC -c -g
	NVCCOPTIONS := -g -arch=sm_30
endif

ifeq ($(MODE), Release)
	CXXOPTIONS := -c -fPIC -O2
	NVCCOPTIONS := -O -arch=sm_30
endif


dist/$(MODE)/$(PLATFORM)/$(TARGET) : $(OCPP_FILES)
	mkdir -p dist/$(MODE)/$(PLATFORM)/
	$(CXX) -shared $(INCLUDE_DIRS) $(OCPP_FILES) -fPIC $(LIBS_DIRS) $(LIBS) -o $@.so
	$(NVCC) $(NVCCOPTIONS) $(NVCC_INCLUDE_DIRS) --cubin $(CU_FILES) -o $@.cubin
build/$(MODE)/$(PLATFORM)/%.o : %.cpp
	mkdir -p build/$(MODE)/$(PLATFORM)/
	$(CXX) $(CXXOPTIONS) $(INCLUDE_DIRS) $< -o $@
clean:
	rm -f dist/$(MODE)/$(PLATFORM)/*
	rm -f build/$(MODE)/$(PLATFORM)/*