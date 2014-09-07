include ../project_generic.mk
TARGET := oglamatrixcputest
INCLUDE_PATHS := 
EXTRA_LIBS := $(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMatrix.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMatrixCpu.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMath.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaUtils.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/libArnoldiPackage.so 
