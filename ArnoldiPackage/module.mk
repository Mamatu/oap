include ../project_generic.mk
TARGET := libArnoldiPackage
INCLUDE_DIRS := -I $(OGLA_PATH)/oglaUtils -I $(OGLA_PATH)/oglaMatrixCuda
EXTRA_LIBS := $(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMatrixCpu.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMatrixCuda.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMath.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaUtils.so 
	
