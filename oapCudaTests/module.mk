include ../project_generic.mk
TARGET := liboapCudaTests
INCLUDE_PATHS :=
EXTRA_LIBS := -L/usr/local/cuda/lib -lcuda\
	$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapCuda.so \