include ../project_generic.mk
TARGET := liboapMatrix
INCLUDE_PATHS :=
EXTRA_LIBS := $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapMath.so\
							$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapUtils.so\
							$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapMemory.so
EXTRA_CXXOPTIONS := -std=c++11

