#include "KernelsOperations.h"

//#define KERNEL_FILE "/home/mmatula/Ogla/oglaMatrixCuda/dist/Debug/GNU-Linux-x86/liboglaMatrixCuda.cubin"

#include <string.h>
#include <linux/fs.h>
#include <cuda_runtime_api.h>
#include "DebugLogs.h"

enum OperationType {
    OPERATION_ADDITION,
    OPERATION_SUBSTRACTION,
    OPERATION_DOT_PRODUCT,
    OPERATION_CONSTANT_MULTIPLICATION,
    OPERATION_EXP,
    OPERATION_DIAGONALIZATION,
    OPERATION_TENSOR_PRODUCT
};

void destroyCtx(void* ptr) {
    CUcontext cuContext = (CUcontext) ptr;
    cuCtxDestroy(cuContext);
}

struct ContextPtrs {
    CUdevice cuDevice;
    CUcontext cuContext;
};

int getSharedMemorySize(OperationType operationType, CUdevprop& cuDevprop) {
    int sharedSize = 0;
    if (operationType == OPERATION_DOT_PRODUCT) {
        sharedSize = cuDevprop.sharedMemPerBlock;
    }
    return sharedSize;
}

#define isRe(m) m->reValues != NULL
#define isIm(m) m->imValues != NULL

namespace math {

    void KernelsOperations::executeKernel(const char* functionName,
            void** params, ::cuda::Kernel& kernel) {
        ::cuda::Kernel::ExecuteKernel(functionName,
                params, kernel, m_image);
    }

    KernelsOperations::KernelsOperations() : utils::Module() {
        this->m_dma = new DeviceMatrixAllocator();
        debugAssert("Not defined path" == NULL);
        m_image = ::cuda::Kernel::LoadImage("");
        if (m_image == NULL) {
            this->addMessageLine("Cubin file was not found.");
        }
    }

    KernelsOperations::~KernelsOperations() {
        delete this->m_dma;
        ::cuda::Kernel::FreeImage(this->m_image);
    }

    void KernelsOperations::dotProductDeviceMatrices(Matrix* output,
            Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel) {
        CONTROL_CRITICALS();
        void* params[] = {&output, &matrix1, &matrix2};
        executeKernel("DotProductKernelReIm", params, kernel);
    }

    void KernelsOperations::dotProductDeviceReMatrices(Matrix* output,
            Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel) {
        CONTROL_CRITICALS();
        void* params[] = {&output, &matrix1, &matrix2};
        executeKernel("DotProductKernelRe", params, kernel);
    }

    void KernelsOperations::dotProductDeviceImMatrices(Matrix* output,
            Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel) {
        CONTROL_CRITICALS();
        void* params[] = {&output, &matrix1, &matrix2};
        executeKernel("DotProductKernelIm", params, kernel);
    }

    void KernelsOperations::addDeviceMatrices(Matrix* output,
            Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel) {
        CONTROL_CRITICALS();
        void* params[] = {&output, &matrix1, &matrix2};
        executeKernel("AddKernelReIm", params, kernel);
    }

    void KernelsOperations::addDeviceReMatrices(Matrix* output,
            Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel) {
        CONTROL_CRITICALS();
        void* params[] = {&output, &matrix1, &matrix2};
        executeKernel("AddKernelRe", params, kernel);
    }

    void KernelsOperations::addDeviceImMatrices(Matrix* output,
            Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel) {
        CONTROL_CRITICALS();
        void* params[] = {&output, &matrix1, &matrix2};
        executeKernel("AddKernelIm", params, kernel);
    }

    void KernelsOperations::substractDeviceMatrices(Matrix* output,
            Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel) {
        CONTROL_CRITICALS();
        void* params[] = {&output, &matrix1, &matrix2};
        executeKernel("SubstractKernel", params, kernel);
    }

    void KernelsOperations::substractDeviceReMatrices(Matrix* output,
            Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel) {
        CONTROL_CRITICALS();
        void* params[] = {&output, &matrix1, &matrix2};
        executeKernel("SubstractKernelRe", params, kernel);
    }

    void KernelsOperations::substractDeviceImMatrices(Matrix* output,
            Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel) {
        CONTROL_CRITICALS();
        void* params[] = {&output, &matrix1, &matrix2};
        executeKernel("SubstractKernelIm", params, kernel);
    }

    void KernelsOperations::multiplyConstantDeviceMatrix(Matrix* output,
            Matrix* matrix1, floatt* value, ::cuda::Kernel& kernel) {
        CONTROL_CRITICALS();
        void* params[] = {&output, &matrix1, &value};
        executeKernel("MultiplyConstantKernelReIm", params, kernel);
    }

    void KernelsOperations::multiplyConstantDeviceReMatrix(Matrix* output,
            Matrix* matrix1, floatt* value, ::cuda::Kernel& kernel) {
        CONTROL_CRITICALS();
        void* params[] = {&output, &matrix1, &value};
        executeKernel("MultiplyConstantKernelRe", params, kernel);
    }

    void KernelsOperations::multiplyConstantDeviceImMatrix(Matrix* output,
            Matrix* matrix1, floatt* value, ::cuda::Kernel& kernel) {
        CONTROL_CRITICALS();
        void* params[] = {&output, &matrix1, &value};
        executeKernel("MultiplyConstantKernelIm", params, kernel);
    }

    void KernelsOperations::expDeviceMatrix(Matrix* output,
            Matrix* matrix1, ::cuda::Kernel& kernel) {
        CONTROL_CRITICALS();
        void* params[] = {&output, &matrix1};
        executeKernel("ExpKernel", params, kernel);
    }

    void KernelsOperations::diagonalizeDeviceMatrix(Matrix* output,
            Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel) {
        CONTROL_CRITICALS();
        void* params[] = {&output, &matrix1, &matrix2};
        executeKernel("DiagonalizationKernel", params, kernel);
    }

    void KernelsOperations::tensorProductDeviceMatrix(Matrix* output,
            Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel) {
        CONTROL_CRITICALS();
        void* params[] = {&output, &matrix1, &matrix2};
        executeKernel("TensorProductKernel", params, kernel);
    }
}