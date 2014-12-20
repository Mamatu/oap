#ifndef OGLA_KERNELS_OPERATIONS_H
#define	OGLA_KERNELS_OPERATIONS_H

#include "KernelExecutor.h"
#include "Module.h"
#include "DeviceMatrixModules.h"

namespace math {

    class KernelsOperations : public utils::Module {
        CUdevprop getDeviceInfo();
        DeviceMatrixAllocator* m_dma;
        void* m_image;
        void executeKernel(const char* functionName, void** params,
                ::cuda::Kernel& kernel);
    public:
        void dotProductDeviceMatrices(Matrix* output,
                Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel);
        void dotProductDeviceReMatrices(Matrix* output,
                Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel);
        void dotProductDeviceImMatrices(Matrix* output,
                Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel);
        void addDeviceMatrices(Matrix* output,
                Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel);
        void addDeviceReMatrices(Matrix* output,
                Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel);
        void addDeviceImMatrices(Matrix* output,
                Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel);
        void substractDeviceMatrices(Matrix* output,
                Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel);
        void substractDeviceReMatrices(Matrix* output,
                Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel);
        void substractDeviceImMatrices(Matrix* output,
                Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel);
        void multiplyConstantDeviceMatrix(Matrix* output,
                Matrix* matrix1, floatt* value, ::cuda::Kernel& kernel);
        void multiplyConstantDeviceReMatrix(Matrix* output,
                Matrix* matrix1, floatt* value, ::cuda::Kernel& kernel);
        void multiplyConstantDeviceImMatrix(Matrix* output,
                Matrix* matrix1, floatt* value, ::cuda::Kernel& kernel);
        void expDeviceMatrix(Matrix* output, Matrix* matrix1,
                ::cuda::Kernel& kernel);
        void diagonalizeDeviceMatrix(Matrix* output,
                Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel);
        void tensorProductDeviceMatrix(Matrix* output,
                Matrix* matrix1, Matrix* matrix2, ::cuda::Kernel& kernel);
        KernelsOperations();
        virtual ~KernelsOperations();
    };
}
#endif	/* MATRIXOPERATIONSCU_H */

