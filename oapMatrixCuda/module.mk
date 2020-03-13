include ../project_generic.mk
TARGET := liboapMatrixCuda
INCLUDE_PATHS :=
EXTRA_LIBS := -L/usr/local/cuda/lib -lcuda\
							$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapCuda.so\
							$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapMemory.so
EXTRA_CXXOPTIONS := -std=c++11

