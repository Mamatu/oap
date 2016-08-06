ifndef OGLA_PATH
$(error OGLA_PATH is not set)
endif

ifndef GTEST_DIR
$(error GTEST_DIR is not set)
endif

ifndef GMOCK_DIR
$(error GMOCK_DIR is not set)
endif


ifeq ($(PROJECT), albert)
include $(OGLA_PATH)/project_albert.mk
else ifeq ($(PROJECT), teslac2050)
include $(OGLA_PATH)/project_teslac2050.mk
else ifeq ($(PROJECT), samsung)
include $(OGLA_PATH)/project_samsung.mk
else ifeq ($(PROJECT), lenovo)
include $(OGLA_PATH)/project_lenovo.mk
else ifeq ($(PROJECT), gpp44)
include $(OGLA_PATH)/project_gpp44.mk
else
include $(OGLA_PATH)/project_lenovo.mk
endif

TYPES := EXTENDED_TYPES

ifeq ($(KERNEL_INFO), 1)
KERNEL_INFO := KERNEL_EXTENDED_INFO=1
else ifeq ($(KERNEL_INFO), 0)
KERNEL_INFO := KERNEL_EXTENDED_INFO=0
else
KERNEL_INFO := KERNEL_EXTENDED_INFO=1
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
OGLA_INCLUDES += ArnoldiPackage
OGLA_INCLUDES += oglaShibataMgr
OGLA_INCLUDES += oglaQRTestSamples
#OGLA_INCLUDES += oglaMatrixCpuTest
#OGLA_INCLUDES += oglaMatrixCudaTest
#OGLA_INCLUDES += oglaParser
#OGLA_INCLUDES += ArnoldiPackageTest
#OGLA_INCLUDES += oglaV3D
OGLA_INCLUDES += oglaTests
OGLA_INCLUDES += oglaTestsHost

OGLA_MODULES := oglaUtils
OGLA_MODULES += oglaServerUtils
OGLA_MODULES += oglaMath
OGLA_MODULES += oglaCuda
OGLA_MODULES += oglaMatrix
OGLA_MODULES += oglaMatrixCpu
OGLA_MODULES += oglaMatrixCuda
OGLA_MODULES += oglaCudaTests
OGLA_MODULES += oglaShibataCpu
OGLA_MODULES += oglaShibataCuda
OGLA_MODULES += ArnoldiPackage
OGLA_MODULES += oglaShibataMgr
#OGLA_MODULES += oglaMatrixCpuTest
#OGLA_MODULES += oglaMatrixCudaTest
#OGLA_MODULES += oglaParser
#OGLA_MODULES += ArnoldiPackageTest
#OGLA_MODULES += oglaV3D
OGLA_MODULES += oglaTestsDevice
OGLA_MODULES += oglaTestsHost

CU_OGLA_MODULES := oglaCuda
CU_OGLA_MODULES += oglaMatrixCuda
CU_OGLA_MODULES += oglaCudaTests
CU_OGLA_MODULES += oglaShibataCuda
CU_OGLA_MODULES += ArnoldiPackage
CU_OGLA_MODULES += oglaShibataMgr
CU_OGLA_MODULES += oglaTestsDevice

