include ../project_generic.mk
TARGET := liboglaCudaTests
INCLUDE_PATHS :=
EXTRA_LIBS := -L/usr/local/cuda/lib -lcuda\
	$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaCuda.so \
