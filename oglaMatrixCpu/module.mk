include ../project_generic.mk
TARGET := liboglaMatrixCpu
INCLUDE_PATHS :=
EXTRA_LIBS := $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMath.so\
	$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaUtils.so \
