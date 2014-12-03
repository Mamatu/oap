include ../project_generic.mk
TARGET := arnoldiTest
INCLUDE_PATHS := -I $(OGLA_PATH)/ArnoldiPackage -I $(OGLA_PATH)/oglaMatrixCpu -I $(OGLA_PATH)/oglaMatrixCuda
	
EXTRA_LIBS := $(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMath.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaUtils.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaCuda.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMatrix.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMatrixCpu.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMatrixCuda.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/libArnoldiPackage.so 
	
