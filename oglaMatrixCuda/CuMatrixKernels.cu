#include "CuMatrixProcedures.h"
#include "CuMatrixUtils.h"

extern "C" __global__ void CUDAKernel_DotProductRe(
        math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyReMatrices(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_DotProductIm(
        math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyImMatrices(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_DotProduct(math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyRealMatrices(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_AddRe(
        math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_addReMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_AddIm(
        math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_addImMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_AddReIm(
        math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_addMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_SubstractRe(math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_substractReMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_SubstractIm(math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_substractImMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_Substract(math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_substractMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_MultiplyConstantRe(math::Matrix* output,
        math::Matrix* params0, floatt* value) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyConstantReMatrix(output, params0, value,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_MultiplyConstantIm(math::Matrix* output,
        math::Matrix* params0, floatt* value) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyConstantImMatrix(output, params0, value,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_MultiplyConstant(math::Matrix* output,
        math::Matrix* params0, floatt* value) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyConstantMatrix(output, params0, value,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_TensorProductRe(math::Matrix* output,
        math::Matrix* params0,
        math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    UDA_tensorProductReMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_TensorProductIm(math::Matrix* output, math::Matrix* params0,
        math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_tensorProductImMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_TensorProduct(math::Matrix* output,
        math::Matrix* params0,
        math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_tensorProductMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_TransposeRe(math::Matrix* output,
        math::Matrix* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_transposeReMatrix(output, params0,
            threadIndexX, threadIndexY);
}
    
extern "C" __global__ void CUDAKernel_TransposeIm(math::Matrix* output,
        math::Matrix* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_transposeImMatrix(output, params0,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_Transpose(math::Matrix* output,
        math::Matrix* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_transposeRealMatrix(output, params0,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_QRRe(math::Matrix* output0,
        math::Matrix* output1,
        math::Matrix* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
}

extern "C" __global__ void CUDAKernel_QRIm(math::Matrix* output0,
        math::Matrix* output1,
        math::Matrix* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
}

extern "C" __global__ void CUDAKernel_QR(math::Matrix* output0,
        math::Matrix* output1,
        math::Matrix* params0) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
}
