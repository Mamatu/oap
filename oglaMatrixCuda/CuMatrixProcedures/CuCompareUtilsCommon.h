#ifndef CUCOMPAREUTILSCOMMON
#define CUCOMPAREUTILSCOMMON

#include "cuda.h"
#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"
#include <math.h>

__hostdevice__ int cuda_isEqualReIndex(math::Matrix* matrix1,
                                       math::Matrix* matrix2, uintt index) {
  CUDA_TEST_INIT();
  return fabs(matrix1->reValues[index] - matrix2->reValues[index]) < 0.0001f
             ? 1
             : 0;
}

__hostdevice__ int cuda_isEqualImIndex(math::Matrix* matrix1,
                                       math::Matrix* matrix2, uintt index) {
  CUDA_TEST_INIT();
  return fabs(matrix1->imValues[index] - matrix2->imValues[index]) < 0.0001f
             ? 1
             : 0;
}

__hostdevice__ int cuda_isEqualRe(math::Matrix* matrix1, math::Matrix* matrix2,
                                  uintt column, uintt row) {
  CUDA_TEST_INIT();
  return fabs(GetRe(matrix1, column, row) - GetRe(matrix2, column, row)) <
                 0.0001f
             ? 1
             : 0;
}

__hostdevice__ int cuda_isEqualIm(math::Matrix* matrix1, math::Matrix* matrix2,
                                  uintt column, uintt row) {
  CUDA_TEST_INIT();
  return fabs(GetIm(matrix1, column, row) - GetIm(matrix2, column, row)) <
                 0.0001f
             ? 1
             : 0;
}

#endif  // CUCOMPAREUTILSCOMMON
