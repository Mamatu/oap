include ../project_generic.mk
TARGET := liboap2dt3dDevice
INCLUDE_PATHS :=
EXTRA_LIBS := $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/libArnoldiPackage.so
EXTRA_CXXOPTIONS := -std=c++11

