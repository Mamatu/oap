ifndef OAP_PATH
$(error OAP_PATH is not set)
endif

ifndef GTEST_DIR
$(error GTEST_DIR is not set)
endif

ifndef GMOCK_DIR
$(error GMOCK_DIR is not set)
endif


ifeq ($(PROJECT), albert)
include $(OAP_PATH)/project_albert.mk
else ifeq ($(PROJECT), teslac2050)
include $(OAP_PATH)/project_teslac2050.mk
else ifeq ($(PROJECT), samsung)
include $(OAP_PATH)/project_samsung.mk
else ifeq ($(PROJECT), lenovo)
include $(OAP_PATH)/project_lenovo.mk
else ifeq ($(PROJECT), gpp44)
include $(OAP_PATH)/project_gpp44.mk
else
include $(OAP_PATH)/project_lenovo.mk
endif

TYPES := EXTENDED_TYPES

ifeq ($(KERNEL_INFO), 1)
KERNEL_INFO := KERNEL_EXTENDED_INFO=1
else ifeq ($(KERNEL_INFO), 0)
KERNEL_INFO := KERNEL_EXTENDED_INFO=0
else
KERNEL_INFO := KERNEL_EXTENDED_INFO=1
endif

OGLA_INCLUDES := oapUtils
OGLA_INCLUDES += oapServerUtils
OGLA_INCLUDES += oapMath
OGLA_INCLUDES += oapCuda
OGLA_INCLUDES += oapMatrix
OGLA_INCLUDES += oapMatrixCpu
OGLA_INCLUDES += oapMatrixCuda
OGLA_INCLUDES += oapCudaTests
OGLA_INCLUDES += oapShibataCpu
OGLA_INCLUDES += oapShibataCuda
OGLA_INCLUDES += ArnoldiPackage
OGLA_INCLUDES += oapShibataMgr
OGLA_INCLUDES += oapQRTestSamples
#OGLA_INCLUDES += oapMatrixCpuTest
#OGLA_INCLUDES += oapMatrixCudaTest
#OGLA_INCLUDES += oapParser
#OGLA_INCLUDES += ArnoldiPackageTest
#OGLA_INCLUDES += oapV3D
OGLA_INCLUDES += oapTests
OGLA_INCLUDES += oapTestsHost

OGLA_MODULES := oapUtils
OGLA_MODULES += oapServerUtils
OGLA_MODULES += oapMath
OGLA_MODULES += oapCuda
OGLA_MODULES += oapMatrix
OGLA_MODULES += oapMatrixCpu
OGLA_MODULES += oapMatrixCuda
OGLA_MODULES += oapCudaTests
OGLA_MODULES += oapShibataCpu
OGLA_MODULES += oapShibataCuda
OGLA_MODULES += ArnoldiPackage
OGLA_MODULES += oapShibataMgr
#OGLA_MODULES += oapMatrixCpuTest
#OGLA_MODULES += oapMatrixCudaTest
#OGLA_MODULES += oapParser
#OGLA_MODULES += ArnoldiPackageTest
#OGLA_MODULES += oapV3D
OGLA_MODULES += oapTestsDevice
OGLA_MODULES += oapTestsHost

CU_OGLA_MODULES := oapCuda
CU_OGLA_MODULES += oapMatrixCuda
CU_OGLA_MODULES += oapCudaTests
CU_OGLA_MODULES += oapShibataCuda
CU_OGLA_MODULES += ArnoldiPackage
CU_OGLA_MODULES += oapShibataMgr
CU_OGLA_MODULES += oapTestsDevice

