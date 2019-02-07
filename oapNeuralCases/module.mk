include ../project_generic.mk
TARGET := liboapNeuralCases
INCLUDE_PATHS :=
EXTRA_CXXOPTIONS := -std=c++11

EXTRA_LIBS := $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapAppUtils.so \
		$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapNeuralDevice.so \
		$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapMatrixCuda.so \
		$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapMatrixCpu.so \
		$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapMatrix.so \
		$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapUtils.so

