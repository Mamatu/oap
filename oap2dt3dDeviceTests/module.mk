include ../project_generic.mk
TARGET := oap2dt3dDeviceTests
INCLUDE_PATHS :=
EXTRA_LIBS := $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboap2dt3dUtils.so \
              $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboap2dt3dDevice.so

#EXTRA_CXXOPTIONS := -std=c++11
