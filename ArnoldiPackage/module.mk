include ../project_generic.mk
TARGET := libArnoldiPackage
INCLUDE_PATHS :=
EXTRA_LIBS := $(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMatrixCpu.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMatrixCuda.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMath.so 
	
