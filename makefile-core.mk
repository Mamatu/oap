include ../project_generic.mk
include module.mk
OGLA_INCLUDES_PATHS := $(addprefix -I$(OAP_PATH)/, $(OGLA_INCLUDES))
CU_FILES := $(wildcard *.cu)
CPP_FILES := $(wildcard *.cpp)
OCPP_FILES := $(addprefix build/$(MODE)/$(PLATFORM)/,$(notdir $(CPP_FILES:.cpp=.o)))
OCU_FILES := $(addprefix build/$(MODE)/$(PLATFORM)/,$(notdir $(CU_FILES:.cu=.o)))
INCLUDE_DIRS := -I/usr/local/cuda/include -I/usr/include $(OGLA_INCLUDES_PATHS)
INCLUDE_DIRS += $(INCLUDE_PATHS)
NVCC_INCLUDE_DIRS := -I/usr/local/cuda/include
LIBS_DIRS := -L/usr/lib/nvidia-current -L/usr/lib -L/usrs/local/cuda/lib64
LIBS := $(EXTRA_LIBS)
SANITIZER_LINKING :=
SANITIZER_COMPILATION :=


ifeq ($(MODE), DebugSanitizer)
        SANITIZER_LINKING := -fsanitize=address -fno-omit-frame-pointer
	SANITIZER_COMPILATION := -fsanitize=address -fno-omit-frame-pointer
	CXXOPTIONS := -c -g3 -D$(TYPES) -D$(KERNEL_INFO) -fPIC
	CXXOPTIONS += $(EXTRA_CXXOPTIONS)
	NVCCOPTIONS := -g -G -D$(TYPES) -D$(KERNEL_INFO)
	NVCCOPTIONS += $(EXTRA_NVCCOPTIONS)
endif

ifeq ($(MODE), Debug)
        SANITIZER_LINKING :=
	SANITIZER_COMPILATION :=
	CXXOPTIONS := -c -g3 -D$(TYPES) -D$(KERNEL_INFO) -fPIC
	CXXOPTIONS += $(EXTRA_CXXOPTIONS)
	NVCCOPTIONS := -g -G -D$(TYPES) -D$(KERNEL_INFO)
	NVCCOPTIONS += $(EXTRA_NVCCOPTIONS)
endif

ifeq ($(MODE), Release)
        SANITIZER_LINKING :=
	SANITIZER_COMPILATION :=
	CXXOPTIONS := -c -O2 -D$(TYPES) -D$(KERNEL_INFO) -fPIC
	CXXOPTIONS += $(EXTRA_CXXOPTIONS)
	NVCCOPTIONS := -O2 -D$(TYPES) -D$(KERNEL_INFO)
	NVCCOPTIONS += $(EXTRA_NVCCOPTIONS)
endif
