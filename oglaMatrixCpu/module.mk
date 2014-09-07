include ../project_generic.mk
TARGET := liboglaMatrixCpu
INCLUDE_PATHS :=
EXTRA_LIBS := $(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMath.so\
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaUtils.so \
