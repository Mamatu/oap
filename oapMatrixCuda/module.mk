include ../project_generic.mk
TARGET := liboapMatrixCuda
INCLUDE_PATHS :=
EXTRA_LIBS := -L/usr/local/cuda/lib -lcuda\
	$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapCuda.so
EXTRA_CXXOPTIONS := -std=c++11

