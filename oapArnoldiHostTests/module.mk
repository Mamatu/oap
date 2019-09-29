include ../project_generic.mk
TARGET := oapArnoldiHostTests
INCLUDE_PATHS :=
EXTRA_LIBS := $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapUtils.so
EXTRA_CXXOPTIONS := -std=c++11
