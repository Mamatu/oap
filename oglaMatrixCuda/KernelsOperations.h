#ifndef OGLA_KERNELS_OPERATIONS_H
#define	OGLA_KERNELS_OPERATIONS_H

#include "KernelExecutor.h"
#include "Module.h"
#include "MatrixStructure.h"
#include "DeviceMatrixModules.h"

namespace math {
    namespace cuda {

        class KernelsOperations : public utils::Module {
            CUdevprop getDeviceInfo();
            DeviceMatrixAllocator* m_dma;
            void* m_image;
            void executeKernel(const char* functionName, void** params,
                    ::cuda::Kernel& kernel);
        public:
            void dotProductDeviceMatrices(MatrixStructure* output,
                    MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel);
            void dotProductDeviceReMatrices(MatrixStructure* output,
                    MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel);
            void dotProductDeviceImMatrices(MatrixStructure* output,
                    MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel);
            void addDeviceMatrices(MatrixStructure* output,
                    MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel);
            void addDeviceReMatrices(MatrixStructure* output,
                    MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel);
            void addDeviceImMatrices(MatrixStructure* output,
                    MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel);
            void substractDeviceMatrices(MatrixStructure* output,
                    MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel);
            void substractDeviceReMatrices(MatrixStructure* output,
                    MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel);
            void substractDeviceImMatrices(MatrixStructure* output,
                    MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel);
            void multiplyConstantDeviceMatrix(MatrixStructure* output,
                    MatrixStructure* matrix1, floatt* value, ::cuda::Kernel& kernel);
            void multiplyConstantDeviceReMatrix(MatrixStructure* output,
                    MatrixStructure* matrix1, floatt* value, ::cuda::Kernel& kernel);
            void multiplyConstantDeviceImMatrix(MatrixStructure* output,
                    MatrixStructure* matrix1, floatt* value, ::cuda::Kernel& kernel);
            void expDeviceMatrix(MatrixStructure* output, MatrixStructure* matrix1,
                    ::cuda::Kernel& kernel);
            void diagonalizeDeviceMatrix(MatrixStructure* output,
                    MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel);
            void tensorProductDeviceMatrix(MatrixStructure* output,
                    MatrixStructure* matrix1, MatrixStructure* matrix2, ::cuda::Kernel& kernel);
            KernelsOperations();
            virtual ~KernelsOperations();
        };
    }
}
#endif	/* MATRIXOPERATIONSCU_H */

