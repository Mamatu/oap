include ../project_generic.mk
TARGET := oap2dt3dFuncTests
INCLUDE_PATHS :=
EXTRA_LIBS := $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapAppUtils.so \
              $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapMatrixCpu.so \
              $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapMatrix.so \
              $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapUtils.so

EXTRA_CXXOPTIONS := -std=c++11
