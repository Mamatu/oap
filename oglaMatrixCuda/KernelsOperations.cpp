#include "KernelsOperations.h"

//#define KERNEL_FILE "/home/mmatula/Ogla/oglaMatrixCuda/dist/Debug/GNU-Linux-x86/liboglaMatrixCuda.cubin"

#define KERNEL_FILE 
#define KERNEL_FILE_1 
const char* kernelsFiles[] = {
    "/home/mmatula/Ogla/oglaMatrixCuda/dist/Debug/GNU-Linux-x86/liboglaMatrixCuda.cubin",
    NULL
};
#include <string.h>
#include <linux/fs.h>
#include <cuda_runtime_api.h>
#include "Types.h"

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
    namespace cuda {

        void KernelsOperations::executeKernel(const char* functionName,
                void** params, ::cuda::Kernel& kernel) {
            debugFuncBegin();
            kernel.setImage(this->m_image);
            kernel.setParams(params);
            kernel.execute(functionName);
            debugFuncEnd();
        }

        KernelsOperations::KernelsOperations() : utils::Module() {
            this->m_dma = new DeviceMatrixAllocator();
            this->m_image = NULL;
            void* image = ::cuda::Kernel::LoadImage(kernelsFiles);
            if (image == NULL) {
                this->addMessageLine("Cubin file was not found.");
            }
            this->m_image = image;
        }

        KernelsOperations::~KernelsOperations() {
            delete this->m_dma;
            ::cuda::Kernel::FreeImage(this->m_image);
        }

        void KernelsOperations::dotProductDeviceMatrices(MatrixStructure* output,
                MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel) {
            CONTROL_CRITICALS();
            void* params[] = {&output, &matrix1, &matrix2};
            executeKernel("DotProductKernelReIm", params, kernel);
        }

        void KernelsOperations::dotProductDeviceReMatrices(MatrixStructure* output,
                MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel) {
            CONTROL_CRITICALS();
            void* params[] = {&output, &matrix1, &matrix2};
            executeKernel("DotProductKernelRe", params, kernel);
        }

        void KernelsOperations::dotProductDeviceImMatrices(MatrixStructure* output,
                MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel) {
            CONTROL_CRITICALS();
            void* params[] = {&output, &matrix1, &matrix2};
            executeKernel("DotProductKernelIm", params, kernel);
        }

        void KernelsOperations::addDeviceMatrices(MatrixStructure* output,
                MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel) {
            CONTROL_CRITICALS();
            void* params[] = {&output, &matrix1, &matrix2};
            executeKernel("AddKernelReIm", params, kernel);
        }

        void KernelsOperations::addDeviceReMatrices(MatrixStructure* output,
                MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel) {
            CONTROL_CRITICALS();
            void* params[] = {&output, &matrix1, &matrix2};
            executeKernel("AddKernelRe", params, kernel);
        }

        void KernelsOperations::addDeviceImMatrices(MatrixStructure* output,
                MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel) {
            CONTROL_CRITICALS();
            void* params[] = {&output, &matrix1, &matrix2};
            executeKernel("AddKernelIm", params, kernel);
        }

        void KernelsOperations::substractDeviceMatrices(MatrixStructure* output,
                MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel) {
            CONTROL_CRITICALS();
            void* params[] = {&output, &matrix1, &matrix2};
            executeKernel("SubstractKernel", params, kernel);
        }

        void KernelsOperations::substractDeviceReMatrices(MatrixStructure* output,
                MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel) {
            CONTROL_CRITICALS();
            void* params[] = {&output, &matrix1, &matrix2};
            executeKernel("SubstractKernelRe", params, kernel);
        }

        void KernelsOperations::substractDeviceImMatrices(MatrixStructure* output,
                MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel) {
            CONTROL_CRITICALS();
            void* params[] = {&output, &matrix1, &matrix2};
            executeKernel("SubstractKernelIm", params, kernel);
        }

        void KernelsOperations::multiplyConstantDeviceMatrix(MatrixStructure* output,
                MatrixStructure* matrix1, floatt* value, ::cuda::Kernel& kernel) {
            CONTROL_CRITICALS();
            void* params[] = {&output, &matrix1, &value};
            executeKernel("MultiplyConstantKernelReIm", params, kernel);
        }

        void KernelsOperations::multiplyConstantDeviceReMatrix(MatrixStructure* output,
                MatrixStructure* matrix1, floatt* value, ::cuda::Kernel& kernel) {
            CONTROL_CRITICALS();
            void* params[] = {&output, &matrix1, &value};
            executeKernel("MultiplyConstantKernelRe", params, kernel);
        }

        void KernelsOperations::multiplyConstantDeviceImMatrix(MatrixStructure* output,
                MatrixStructure* matrix1, floatt* value, ::cuda::Kernel& kernel) {
            CONTROL_CRITICALS();
            void* params[] = {&output, &matrix1, &value};
            executeKernel("MultiplyConstantKernelIm", params, kernel);
        }

        void KernelsOperations::expDeviceMatrix(MatrixStructure* output,
                MatrixStructure* matrix1, ::cuda::Kernel& kernel) {
            CONTROL_CRITICALS();
            void* params[] = {&output, &matrix1};
            executeKernel("ExpKernel", params, kernel);
        }

        void KernelsOperations::diagonalizeDeviceMatrix(MatrixStructure* output,
                MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel) {
            CONTROL_CRITICALS();
            void* params[] = {&output, &matrix1, &matrix2};
            executeKernel("DiagonalizationKernel", params, kernel);
        }

        void KernelsOperations::tensorProductDeviceMatrix(MatrixStructure* output,
                MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel) {
            CONTROL_CRITICALS();
            void* params[] = {&output, &matrix1, &matrix2};
            executeKernel("TensorProductKernel", params, kernel);
        }
    }
}