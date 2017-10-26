include ../project_generic.mk
TARGET := liboapMatrixCpu
INCLUDE_PATHS :=
EXTRA_LIBS := $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapMath.so\
	$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapUtils.so
EXTRA_CXXOPTIONS := -std=c++11
