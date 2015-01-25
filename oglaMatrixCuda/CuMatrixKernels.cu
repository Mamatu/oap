#include "CuMatrixProcedures.h"
#include "CuMatrixUtils.h"

extern "C" __global__ void CUDAKernel_DotProductRe(
        math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    cuda_debug_function();
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
    CUDA_multiplyMatrices(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_DotProductReExp(
        math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1, MatrixEx* matrixEx) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyReMatricesEx(output, params0, params1, *matrixEx,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_DotProductImExp(
        math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1, MatrixEx* matrixEx) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyImMatricesEx(output, params0, params1, *matrixEx,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_DotProductEx(math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1, MatrixEx* matrixEx) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyMatricesEx(output, params0, params1, *matrixEx,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_AddRe(
        math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_addReMatrices(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_AddIm(
        math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_addImMatrices(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_Add(
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
    CUDA_substractReMatrices(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_SubstractIm(math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_substractImMatrices(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_SubstractReal(math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_substractRealMatrices(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_Substract(math::Matrix* output,
        math::Matrix* params0, math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_substractMatrices(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_MultiplyConstantRe(math::Matrix* output,
        math::Matrix* params0, floatt re) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyConstantReMatrix(output, params0, re,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_MultiplyConstantIm(math::Matrix* output,
        math::Matrix* params0, floatt im) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyConstantImMatrix(output, params0, im,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_MultiplyConstant(math::Matrix* output,
        math::Matrix* params0, floatt re, floatt im) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyConstantMatrix(output, params0, re, im,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_MultiplyConstantReal(math::Matrix* output,
        math::Matrix* params0, floatt re, floatt im) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_multiplyConstantRealMatrix(output, params0, re, im,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_TensorProductRe(math::Matrix* output,
        math::Matrix* params0,
        math::Matrix* params1) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_tensorProductReMatrix(output, params0, params1,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_TensorProductIm(math::Matrix* output,
        math::Matrix* params0,
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
    CUDA_transposeMatrix(output, params0,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_TransposeEx(math::Matrix* output,
        math::Matrix* params0, MatrixEx* matrixEx) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_transposeMatrixEx(output, params0, *matrixEx,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_Magnitude(floatt* value,
        math::Matrix* params0, floatt* buffer) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_magnitude(*value, params0, buffer,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_QRRe(math::Matrix* output0,
        math::Matrix* output1,
        math::Matrix* params0,
        math::Matrix* aux0, math::Matrix* aux1,
        math::Matrix* aux2, math::Matrix* aux3) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    //CUDA_QRRe(output0, output1, params0, aux0, aux1, aux2, aux3,
      //      threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_QRIm(math::Matrix* output0,
        math::Matrix* output1,
        math::Matrix* params0,
        math::Matrix* aux0, math::Matrix* aux1,
        math::Matrix* aux2, math::Matrix* aux3) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    //CUDA_QRIm(output0, output1, params0, aux0, aux1, aux2, aux3,
    //        threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_QR(math::Matrix* output0,
        math::Matrix* output1,
        math::Matrix* params0,
        math::Matrix* aux0, math::Matrix* aux1,
        math::Matrix* aux2, math::Matrix* aux3) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_QR(output0, output1, params0, aux0, aux1, aux2, aux3,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_SetVector(math::Matrix* output, uintt index,
        math::Matrix* params0, uintt length) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_setVector(output, index, params0, length, threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_GetVector(math::Matrix* output, uintt length,
        math::Matrix* params0, uintt index) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_getVector(output, length, params0, index, threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_SetIdentity(math::Matrix* matrix) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_setIdentityMatrix(matrix, threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_SetDiagonal(math::Matrix* matrix,
        floatt re, floatt im) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_setDiagonalMatrix(matrix, re, im, threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_CompareRe(int* sums,
        math::Matrix* matrix1,
        math::Matrix* matrix2,
        int* buffer) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_compareReMatrix(sums, matrix1, matrix2, buffer,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_Compare(int* sums,
        math::Matrix* matrix1,
        math::Matrix* matrix2,
        int* buffer) {
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_compare(sums, matrix1, matrix2, buffer,
            threadIndexX, threadIndexY);
}

extern "C" __global__ void CUDAKernel_CompareOpt(int* sums,
        math::Matrix* matrix1,
        math::Matrix* matrix2) {
    extern __shared__ int shBuffer[];
    uintt threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    uintt threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
    CUDA_compare(sums, matrix1, matrix2, shBuffer,
            threadIndexX, threadIndexY);
}
