OAP_PATH := $(PWD)
TMP_PATH := /tmp/Oap

ifndef OAP_GMOCK_PATH
$(error OAP_GMOCK_PATH is not set (should be main directory of gmock set))
endif

GTEST_DIR := $(OAP_GMOCK_PATH)/gtest

$(shell test -d /tmp/Oap/tests_data || mkdir -p /tmp/Oap/tests_data)
$(shell test -d /tmp/Oap/conversion_data || mkdir -p /tmp/Oap/conversion_data)
$(shell test -d /tmp/Oap/host_tests || mkdir -p /tmp/Oap/host_tests)
$(shell test -d /tmp/Oap/device_tests || mkdir -p /tmp/Oap/device_tests)

ifeq ($(PROJECT), albert)
include $(OAP_PATH)/project_albert.mk
else ifeq ($(PROJECT), teslac2050)
include $(OAP_PATH)/project_teslac2050.mk
else ifeq ($(PROJECT), samsung)
include $(OAP_PATH)/project_samsung.mk
else ifeq ($(PROJECT), x86)
include $(OAP_PATH)/project_x86.mk
else ifeq ($(PROJECT), gpp44)
include $(OAP_PATH)/project_gpp44.mk
else
include $(OAP_PATH)/project_x86.mk
endif

TYPES := OAP_CONFIG_NI_EF

ifeq ($(KERNEL_INFO), 1)
KERNEL_INFO := KERNEL_EXTENDED_INFO=1
else ifeq ($(KERNEL_INFO), 0)
KERNEL_INFO := KERNEL_EXTENDED_INFO=0
else
KERNEL_INFO := KERNEL_EXTENDED_INFO=0
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
OAP_INCLUDES += oapHostTests
OAP_INCLUDES += oapAppUtils
OAP_INCLUDES += oap2dt3dDevice

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
OAP_MODULES += oapHostTests
endif

ifeq ($(COMPILE_DEVICE),1)
OAP_MODULES += oapCuda
OAP_MODULES += oapMatrixCuda
OAP_MODULES += oapCudaTests
OAP_MODULES += ArnoldiPackage
OAP_MODULES += oapDeviceTests
endif

ifeq ($(COMPILE_HOST),1)
OAP_MODULES += oapAppUtils
OAP_MODULES += oap2dt3dTests
OAP_MODULES += oap2dt3dFuncTests
endif

ifeq ($(COMPILE_DEVICE),1)
OAP_MODULES += oap2dt3dDevice
OAP_MODULES += oap2dt3dDeviceTests
OAP_MODULES += oap2dt3d
endif

CU_OAP_MODULES := oapCuda
CU_OAP_MODULES += oapMatrixCuda
CU_OAP_MODULES += oapCudaTests
CU_OAP_MODULES += oapShibataCuda
CU_OAP_MODULES += ArnoldiPackage
CU_OAP_MODULES += oapShibataMgr
CU_OAP_MODULES += oapDeviceTests
