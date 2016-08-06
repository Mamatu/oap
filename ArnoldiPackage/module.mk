include ../project_generic.mk
TARGET := libArnoldiPackage
INCLUDE_DIRS := -I$(OAP_PATH)/oglaUtils \
	-I$(OAP_PATH)/oglaMatrixCuda \
	-I$(OAP_PATH)/oglaMatrixCpu
	
EXTRA_LIBS := $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMatrixCpu.so \
	$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMatrixCuda.so \
	$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMath.so \
	$(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaUtils.so 
