include ../project_generic.mk
TARGET := oapNeuralDeviceTests
INCLUDE_PATHS :=
EXTRA_LIBS := $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapUtils.so \
							$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapNeuralDevice.so \
							$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapNeuralRoutines.so
EXTRA_CXXOPTIONS := -std=c++11
