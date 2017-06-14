include ../project_generic.mk

$(shell tar -zxzf ./data/images_monkey.tar.gz -C ./data)

TARGET := oap2dt3d
INCLUDE_PATHS :=
EXTRA_LIBS := $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboap2dt3dUtils.so \
              $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapMatrixCpu.so \
              $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapMatrix.so \
              $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapUtils.so

EXTRA_CXXOPTIONS := -std=c++11
