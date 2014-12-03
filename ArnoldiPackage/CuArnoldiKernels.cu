#include "CuArnodliProcedures.h"
#include "CuMatrixUtils.h"
#include "Types.h"

extern "C" __global__ void CUDAKernel_CalculateH(MatrixStructure* Q,
        MatrixStructure* temp, MatrixStructure* temp1,
        MatrixStructure* R, MatrixStructure* H,
        MatrixStructure* temp2, MatrixStructure* temp3,
        MatrixStructure* temp4, MatrixStructure* temp5) {
    CUDA_DEBUG();
    PRINT("HAAA",H);
    CUDA_CalculateH(Q, temp, temp1, R, H, temp2, temp3, temp4, temp5);
}

extern "C" __global__ void CUDAKernel_Execute(bool init, intt initj,
        Matrices* matrices) {
    CUDA_DEBUG();
    CUDA_execute(init, initj, matrices);
}