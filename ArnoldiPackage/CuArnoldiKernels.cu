#include "CuArnodliProceduresImpl.h"
#include "CuMatrixUtils.h"
#include "Types.h"

extern "C" __global__ void CUDAKernel_CalculateTriangularH(math::Matrix* H,
        math::Matrix* Q, math::Matrix* R,
        math::Matrix* temp, math::Matrix* temp1,
        math::Matrix* temp2, math::Matrix* temp3,
        math::Matrix* temp4, math::Matrix* temp5) {
    CUDA_DEBUG();
    CUDA_CalculateTriangularH(H, Q, R, temp, temp1, temp2, temp3, temp4, temp5);
}

extern "C" __global__ void CUDAKernel_CalculateH(bool init, intt initj,
        Matrices* matrices) {
    CUDA_DEBUG();
    CUDA_CalculateH(init, initj, matrices);
}