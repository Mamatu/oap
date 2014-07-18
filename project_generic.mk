OGLA_PATH = /home/mmatula/Ogla
ifeq ($(PROJECT), albert)
include $(OGLA_PATH)/project_albert.mk
endif

ifeq ($(PROJECT), samsung)
include $(OGLA_PATH)/project_samsung.mk
endif

OGLA_INCLUDES := oglaUtils
OGLA_INCLUDES += oglaServerUtils
OGLA_INCLUDES += oglaMath
OGLA_INCLUDES += oglaCuda
OGLA_INCLUDES += oglaMatrix
OGLA_INCLUDES += oglaMatrixCpu
OGLA_INCLUDES += oglaMatrixCuda
OGLA_INCLUDES += oglaShibataCpu
OGLA_INCLUDES += oglaShibataCuda
OGLA_INCLUDES += oglaMatrixCpuTest
OGLA_INCLUDES += oglaMatrixCudaTest
#OGLA_INCLUDES += oglaShibataMgr
