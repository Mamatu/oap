#include "CuMatrixProcedures.h"
#include "CuMatrixUtils.h"

extern "C" __global__ void CUDAKernel_DotProductRe(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1) {
  CUDA_dotProductRe(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_DotProductIm(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1) {
  CUDA_dotProductIm(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_DotProduct(math::Matrix* output,
                                                 math::Matrix* params0,
                                                 math::Matrix* params1) {
  CUDA_dotProduct(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_DotProductReExp(math::Matrix* output,
                                                      math::Matrix* params0,
                                                      math::Matrix* params1,
                                                      MatrixEx* matrixEx) {
  CUDA_dotProductReEx(output, params0, params1, *matrixEx);
}

extern "C" __global__ void CUDAKernel_DotProductImExp(math::Matrix* output,
                                                      math::Matrix* params0,
                                                      math::Matrix* params1,
                                                      MatrixEx* matrixEx) {
  CUDA_dotProductImEx(output, params0, params1, *matrixEx);
}

extern "C" __global__ void CUDAKernel_DotProductEx(math::Matrix* output,
                                                   math::Matrix* params0,
                                                   math::Matrix* params1,
                                                   MatrixEx* matrixEx) {
  CUDA_dotProductEx(output, params0, params1, *matrixEx);
}

extern "C" __global__ void CUDAKernel_DotProductReOpt(math::Matrix* output,
                                                      math::Matrix* params0,
                                                      math::Matrix* params1) {
  extern __shared__ floatt bufferFloat[];
  CUDA_dotProductReOpt(output, params0, params1, bufferFloat);
}

extern "C" __global__ void CUDAKernel_DotProductImOpt(math::Matrix* output,
                                                      math::Matrix* params0,
                                                      math::Matrix* params1) {
  extern __shared__ floatt bufferFloat[];
  CUDA_dotProductImOpt(output, params0, params1, bufferFloat);
}

extern "C" __global__ void CUDAKernel_DotProductOpt(math::Matrix* output,
                                                    math::Matrix* params0,
                                                    math::Matrix* params1) {
  extern __shared__ floatt bufferFloat[];
  CUDA_dotProductOpt(output, params0, params1, bufferFloat);
}

extern "C" __global__ void CUDAKernel_DotProductReExpOpt(math::Matrix* output,
                                                         math::Matrix* params0,
                                                         math::Matrix* params1,
                                                         MatrixEx* matrixEx) {
  extern __shared__ floatt bufferFloat[];
  CUDA_dotProductReExOpt(output, params0, params1, *matrixEx, bufferFloat);
}

extern "C" __global__ void CUDAKernel_DotProductImExpOpt(math::Matrix* output,
                                                         math::Matrix* params0,
                                                         math::Matrix* params1,
                                                         MatrixEx* matrixEx) {
  extern __shared__ floatt bufferFloat[];
  CUDA_dotProductImExOpt(output, params0, params1, *matrixEx, bufferFloat);
}

extern "C" __global__ void CUDAKernel_DotProductExOpt(math::Matrix* output,
                                                      math::Matrix* params0,
                                                      math::Matrix* params1,
                                                      MatrixEx* matrixEx) {
  extern __shared__ floatt bufferFloat[];
  CUDA_dotProductExOpt(output, params0, params1, *matrixEx, bufferFloat);
}

extern "C" __global__ void CUDAKernel_AddRe(math::Matrix* output,
                                            math::Matrix* params0,
                                            math::Matrix* params1) {
  CUDA_addReMatrices(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_AddIm(math::Matrix* output,
                                            math::Matrix* params0,
                                            math::Matrix* params1) {
  CUDA_addImMatrices(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_Add(math::Matrix* output,
                                          math::Matrix* params0,
                                          math::Matrix* params1) {
  CUDA_addMatrix(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_SubstractRe(math::Matrix* output,
                                                  math::Matrix* params0,
                                                  math::Matrix* params1) {
  CUDA_substractReMatrices(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_SubstractIm(math::Matrix* output,
                                                  math::Matrix* params0,
                                                  math::Matrix* params1) {
  CUDA_substractImMatrices(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_SubstractReal(math::Matrix* output,
                                                    math::Matrix* params0,
                                                    math::Matrix* params1) {
  CUDA_substractRealMatrices(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_Substract(math::Matrix* output,
                                                math::Matrix* params0,
                                                math::Matrix* params1) {
  CUDA_substractMatrices(output, params0, params1);
}

extern "C" __global__ void CUDAKernel_MultiplyConstantRe(math::Matrix* output,
                                                         math::Matrix* params0,
                                                         floatt re) {
  CUDA_multiplyConstantReMatrix(output, params0, re);
}

extern "C" __global__ void CUDAKernel_MultiplyConstantIm(math::Matrix* output,
                                                         math::Matrix* params0,
                                                         floatt im) {
  CUDA_multiplyConstantImMatrix(output, params0, im);
}

extern "C" __global__ void CUDAKernel_MultiplyConstant(math::Matrix* output,
                                                       math::Matrix* params0,
                                                       floatt re, floatt im) {
  CUDA_multiplyConstantMatrix(output, params0, re, im);
}

extern "C" __global__ void CUDAKernel_MultiplyConstantReal(
    math::Matrix* output, math::Matrix* params0, floatt re, floatt im) {
  CUDA_multiplyConstantRealMatrix(output, params0, re, im);
}

extern "C" __global__ void CUDAKernel_TransposeRe(math::Matrix* output,
                                                  math::Matrix* params0) {
  CUDA_transposeReMatrix(output, params0);
}

extern "C" __global__ void CUDAKernel_TransposeIm(math::Matrix* output,
                                                  math::Matrix* params0) {
  CUDA_transposeImMatrix(output, params0);
}

extern "C" __global__ void CUDAKernel_Transpose(math::Matrix* output,
                                                math::Matrix* params0) {
  CUDA_transposeMatrix(output, params0);
}

extern "C" __global__ void CUDAKernel_TransposeEx(math::Matrix* output,
                                                  math::Matrix* params0,
                                                  MatrixEx* matrixEx) {
  CUDA_transposeMatrixEx(output, params0, *matrixEx);
}

extern "C" __global__ void CUDAKernel_Magnitude(floatt* value,
                                                math::Matrix* params0,
                                                floatt* buffer) {
  CUDA_magnitudeOpt(value, params0, buffer);
}

extern "C" __global__ void CUDAKernel_QRGR(
    math::Matrix* output0, math::Matrix* output1, math::Matrix* params0,
    math::Matrix* aux0, math::Matrix* aux1, math::Matrix* aux2,
    math::Matrix* aux3) {
  CUDA_QRGR(output0, output1, params0, aux0, aux1, aux2, aux3);
}

extern "C" __global__ void CUDAKernel_SetVector(math::Matrix* output,
                                                uintt index,
                                                math::Matrix* params0,
                                                uintt length) {
  CUDA_setVector(output, index, params0, length);
}

extern "C" __global__ void CUDAKernel_GetVector(math::Matrix* output,
                                                uintt length,
                                                math::Matrix* params0,
                                                uintt index) {
  CUDA_getVector(output, length, params0, index);
}

extern "C" __global__ void CUDAKernel_SetIdentity(math::Matrix* matrix) {
  CUDA_setIdentityMatrix(matrix);
}

extern "C" __global__ void CUDAKernel_SetDiagonal(math::Matrix* matrix,
                                                  floatt re, floatt im) {
  CUDA_setDiagonalMatrix(matrix, re, im);
}

extern "C" __global__ void CUDAKernel_Invert(math::Matrix* output,
                                             math::Matrix* matrix,
                                             math::Matrix* aux1,
                                             math::Matrix* aux2,
                                             math::Matrix* aux3) {
  CUDA_invertMatrix(output, matrix, aux1, aux2, aux3);
}

extern "C" __global__ void CUDAKernel_CompareRe(int* sums,
                                                math::Matrix* matrix1,
                                                math::Matrix* matrix2,
                                                int* buffer,
                                                uintt bufferLength) {
  CUDA_compareReMatrix(sums, matrix1, matrix2, buffer);
}

extern "C" __global__ void CUDAKernel_Compare(int* sums, math::Matrix* matrix1,
                                              math::Matrix* matrix2,
                                              int* buffer, uintt bufferLength) {
  CUDA_compare(sums, matrix1, matrix2, buffer);
}

extern "C" __global__ void CUDAKernel_CompareOpt(int* sums,
                                                 math::Matrix* matrix1,
                                                 math::Matrix* matrix2) {
  extern __shared__ int sharedBufferInt[];
  CUDA_compareOpt(sums, matrix1, matrix2, sharedBufferInt);
}

extern "C" __global__ void CUDAKernel_CompareOptVer2(int* sums,
                                                     math::Matrix* matrix1,
                                                     math::Matrix* matrix2) {
  extern __shared__ int sharedBufferInt[];
  CUDA_compareOptVer2(sums, matrix1, matrix2, sharedBufferInt);
}

extern "C" __global__ void CUDAKernel_MagnitudeOpt(floatt* sums,
                                                   math::Matrix* matrix) {
  extern __shared__ floatt bufferFloat[];
  CUDA_magnitudeOpt(sums, matrix, bufferFloat);
}

extern "C" __global__ void CUDAKernel_MagnitudeOptVer2(floatt* sums,
                                                       math::Matrix* matrix) {
  extern __shared__ floatt bufferFloat[];
  CUDA_magnitudeOptVer2(sums, matrix, bufferFloat);
}

extern "C" __global__ void CUDAKernel_IsUpperTriangular(int* outcome,
                                                        math::Matrix* matrix) {
  int is = CUDA_isUpperTriangular(matrix);
  (*outcome) = is;
}

extern "C" __global__ void CUDAKernel_CalculateTriangularH(
    math::Matrix* H, math::Matrix* Q, math::Matrix* R, math::Matrix* temp,
    math::Matrix* temp1, math::Matrix* temp2, math::Matrix* temp3,
    math::Matrix* temp4, math::Matrix* temp5) {
  CUDA_HMtoUTM(H, Q, R, temp, temp1, temp2, temp3, temp4, temp5);
}
