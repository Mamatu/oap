#
CPP_FILES = $(wildcard *.cpp)
CU_FILES = $(wildcard *.cu)
MODE := Debug
OCPP_FILES := $(addprefix build/$(MODE)/,$(notdir $(CPP_FILES:.cpp=.o)))
PLATFORM = 
OCU_FILES := $(addprefix build/$(MODE)/,$(notdir $(CU_FILES:.cu=.o)))

OGLA_PATH = /home/mmatula/Ogla

TARGET := libGACuda.so
CXX := g++
NVCC := nvcc
OGLA_PROJECTS_INCLUDE_DIRS = -I$(OGLA_PATH)/oglaUtils -I$(OGLA_PATH)/oglaGA -I$(OGLA_PATH)/oglaServer -I$(OGLA_PATH)/oglaMatrix
INCLUDE_DIRS := -I/usr/local/cuda/include -I/usr/include -I/usr/X11R6/include
INCLUDE_DIRS += $(OGLA_PROJECTS_INCLUDE_DIRS )
CUDA_INCLUDES := 
LIBS_DIRS := -L/usr/lib/nvidia-current  -L/usr/lib -L/usr/local/lib -L/usr/local/cuda/lib
LIBS :=  /usr/lib/libgsl.a /usr/lib/libgsl.so

ifeq ($(MODE), Debug)
	CXXOPTIONS := -c -g
	NVCCOPTIONS := -ldl -c -g -G
endif

ifeq ($(MODE), Release)
	CXXOPTIONS := -c -O3
	NVCCOPTIONS := -c -O
endif

dist/$(MODE)/$(TARGET) : $(OCU_FILES) $(OCPP_FILES)
	$(CXX) -shared $(OCU_FILES) $(OCPP_FILES) -fPIC $(LIBS) $(LIBS_DIRS) 
build/$(MODE)/%.o : %.cpp
	$(CXX) $(CXXOPTIONS) $(INCLUDE_DIRS) $< -o $@
build/$(MODE)/%.o : %.cu
	$(NVCC) $(NVCCOPTIONS) $(CUDA_INCLUDES) $< -o $@
clean:
	rm -f dist/debug/$(TARGET)
	rm -f dist/release/$(TARGET)
	rm -f build/release/*
	rm -f build/debug/*


