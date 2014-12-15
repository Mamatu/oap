#include "CuArnodliProceduresImpl.h"
#include "CuMatrixUtils.h"
#include "Types.h"

extern "C" __global__ void CUDAKernel_CalculateHQ(MatrixStructure* H,
        MatrixStructure* Q, MatrixStructure* R,
        MatrixStructure* temp, MatrixStructure* temp1,
        MatrixStructure* temp2, MatrixStructure* temp3,
        MatrixStructure* temp4, MatrixStructure* temp5) {
    CUDA_DEBUG();
    cuda_debug_structure("Kernel_H", H);
    cuda_debug_structure("Kernel_Q", Q);
    cuda_debug_structure("Kernel_R", R);
    CUDA_CalculateHQ(H, Q, R, temp, temp1, temp2, temp3, temp4, temp5);
}

extern "C" __global__ void CUDAKernel_Execute(bool init, intt initj,
        Matrices* matrices) {
    CUDA_DEBUG();
    CUDA_execute(init, initj, matrices);
}