include ../project_generic.mk
TARGET := oapDeviceTests
INCLUDE_PATHS :=
EXTRA_LIBS := $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapUtils.so
EXTRA_CXXOPTIONS := -std=c++11
