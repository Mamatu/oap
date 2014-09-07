include ../project_generic.mk
TARGET := oglamatrixcudatest
INCLUDE_PATHS := 
EXTRA_LIBS := -L/usr/local/cuda-6.0/lib -lcudart -lcuda \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMatrix.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMatrixCpu.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMatrixCuda.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaMath.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/liboglaUtils.so \
	$(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/libArnoldiPackage.so 
