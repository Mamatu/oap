include ../project_generic.mk
TARGET := oapAppUtilsTests
INCLUDE_PATHS :=
EXTRA_LIBS := $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapAppUtils.so
EXTRA_CXXOPTIONS := -std=c++11
