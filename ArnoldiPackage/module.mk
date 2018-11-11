include ../project_generic.mk
TARGET := libArnoldiPackage
INCLUDE_DIRS := -I$(OAP_PATH)/oapUtils \
	-I$(OAP_PATH)/oapMatrixCuda \
	-I$(OAP_PATH)/oapMatrixCpu
	
EXTRA_LIBS := $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapMatrixCpu.so \
	$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapMatrixCuda.so \
	$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapMath.so \
	$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboapUtils.so 

EXTRA_CXXOPTIONS := -std=c++11
