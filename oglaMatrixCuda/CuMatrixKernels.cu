#include "CuMatrixProcedures.h"
#include "CuMatrixUtils.h"

extern "C" __global__ void CUDAKernel_DotProductRe(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyReMatrices(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_DotProductIm(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyImMatrices(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_DotProduct(MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyMatrices(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_AddRe(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_addReMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_AddIm(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_addImMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_AddReIm(
        MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_addMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_SubstractRe(MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_substractReMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_SubstractIm(MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_substractImMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_Substract(MatrixStructure* output,
        MatrixStructure* params0, MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_substractMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_MultiplyConstantRe(MatrixStructure* output,
        MatrixStructure* params0, floatt* value) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyConstantReMatrix(output, params0, value,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_MultiplyConstantIm(MatrixStructure* output,
        MatrixStructure* params0, floatt* value) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyConstantImMatrix(output, params0, value,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_MultiplyConstant(MatrixStructure* output,
        MatrixStructure* params0, floatt* value) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyConstantMatrix(output, params0, value,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_TensorProductRe(MatrixStructure* output,
        MatrixStructure* params0,
        MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    UDA_tensorProductReMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_TensorProductIm(MatrixStructure* output, MatrixStructure* params0,
        MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_tensorProductImMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_TensorProduct(MatrixStructure* output,
        MatrixStructure* params0,
        MatrixStructure* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_tensorProductMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_TransposeRe(MatrixStructure* output,
        MatrixStructure* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_transposeReMatrix(output, params0,
            threadIndexX, threadIndexY);
}
    
extern "C" __global__ void CUDAKernel_TransposeIm(MatrixStructure* output,
        MatrixStructure* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_transposeImMatrix(output, params0,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_Transpose(MatrixStructure* output,
        MatrixStructure* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_transposeMatrix(output, params0,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_QRRe(MatrixStructure* output0,
        MatrixStructure* output1,
        MatrixStructure* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
}

extern "C" __global__ void CUDAKernel_QRIm(MatrixStructure* output0,
        MatrixStructure* output1,
        MatrixStructure* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
}

extern "C" __global__ void CUDAKernel_QR(MatrixStructure* output0,
        MatrixStructure* output1,
        MatrixStructure* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
}
