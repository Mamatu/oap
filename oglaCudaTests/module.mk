include ../project_generic.mk
TARGET := liboglaCudaTests
INCLUDE_PATHS := -lcudart -lcuda
EXTRA_LIBS := -L/usr/local/cuda-6.0/lib -lcudart -lcuda\
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaCuda.so \
