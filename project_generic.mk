OGLA_PATH = /home/mmatula/Ogla
ifeq ($(PROJECT), albert)
include $(OGLA_PATH)/project_albert.mk
else ifeq ($(PROJECT), samsung)
include $(OGLA_PATH)/project_samsung.mk
else
include $(OGLA_PATH)/project_albert.mk
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
OGLA_INCLUDES += ArnoldiPackage
OGLA_INCLUDES += oglaMatrixCpuTest
OGLA_INCLUDES += oglaMatrixCudaTest
#OGLA_INCLUDES += oglaParser
OGLA_INCLUDES += ArnoldiPackageTest
#OGLA_INCLUDES += oglaShibataMgr
