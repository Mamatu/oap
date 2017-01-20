ifndef OAP_PATH
$(error OAP_PATH is not set)
endif

ifndef GMOCK_DIR
$(error GMOCK_DIR is not set (should be main directory of gmock set))
endif

GTEST_DIR := $(GMOCK_DIR)/gtest


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

OAP_INCLUDES := oapUtils
OAP_INCLUDES += oapMath
OAP_INCLUDES += oapCuda
OAP_INCLUDES += oapMatrix
OAP_INCLUDES += oapMatrixCpu
OAP_INCLUDES += oapMatrixCuda
OAP_INCLUDES += oapCudaTests
OAP_INCLUDES += ArnoldiPackage
OAP_INCLUDES += oapQRTestSamples
OAP_INCLUDES += oapTests
OAP_INCLUDES += oapTestsHost
OAP_INCLUDES += oap2dt3dUtils

TARGET_ARCH := DEVICE_HOST

COMPILE_HOST := 0
COMPILE_DEVICE := 0

ifeq ($(TARGET_ARCH),DEVICE_HOST)
COMPILE_DEVICE := 1
COMPILE_HOST := 1
endif

ifeq ($(TARGET_ARCH),DEVICE)
COMPILE_DEVICE := 1
endif

ifeq ($(TARGET_ARCH),HOST)
COMPILE_HOST := 1
endif


ifeq ($(COMPILE_HOST),1)
OAP_MODULES := oapUtils
OAP_MODULES += oapMath
OAP_MODULES += oapMatrix
OAP_MODULES += oapMatrixCpu
OAP_MODULES += oapTestsHost
endif

ifeq ($(COMPILE_DEVICE),1)
OAP_MODULES += oapCuda
OAP_MODULES += oapMatrixCuda
OAP_MODULES += oapCudaTests
OAP_MODULES += ArnoldiPackage
OAP_MODULES += oapTestsDevice
endif

ifeq ($(COMPILE_HOST),1)
OAP_MODULES += oap2dt3dUtils
OAP_MODULES += oap2dt3d
OAP_MODULES += oap2dt3dTests
OAP_MODULES += oap2dt3dFuncTests
endif

ifeq ($(COMPILE_DEVICE),1)
OAP_MODULES += oap2dt3dDevice
endif

CU_OAP_MODULES := oapCuda
CU_OAP_MODULES += oapMatrixCuda
CU_OAP_MODULES += oapCudaTests
CU_OAP_MODULES += oapShibataCuda
CU_OAP_MODULES += ArnoldiPackage
CU_OAP_MODULES += oapShibataMgr
CU_OAP_MODULES += oapTestsDevice
