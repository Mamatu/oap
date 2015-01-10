OGLA_PATH = /home/mmatula/Ogla
GTEST_DIR = /home/mmatula/Ogla/gmock-1.7.0/gtest
GMOCK_DIR = /home/mmatula/Ogla/gmock-1.7.0
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
OGLA_INCLUDES += oglaCudaTests
OGLA_INCLUDES += oglaShibataCpu
OGLA_INCLUDES += oglaShibataCuda
#OGLA_INCLUDES += oglaShibataMgr
OGLA_INCLUDES += ArnoldiPackage
OGLA_INCLUDES += oglaMatrixCpuTest
#OGLA_INCLUDES += oglaMatrixCudaTest
#OGLA_INCLUDES += oglaParser
OGLA_INCLUDES += ArnoldiPackageTest
OGLA_INCLUDES += oglaV3D
OGLA_INCLUDES += oglaTests
