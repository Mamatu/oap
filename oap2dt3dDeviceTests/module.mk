include ../project_generic.mk
TARGET := oap2dt3dDeviceTests
INCLUDE_PATHS :=
EXTRA_LIBS := $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapAppUtils.so \
              $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboap2dt3dDevice.so \
              $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapMatrixCuda.so
EXTRA_CXXOPTIONS := -std=c++11
