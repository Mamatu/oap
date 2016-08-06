#ifndef CUCOMPAREUTILSCOMMON
#define CUCOMPAREUTILSCOMMON

#include "cuda.h"
#include "CuCore.h"
#include "Matrix.h"
#include "MatrixAPI.h"
#include <math.h>

#define COMPARE_LIMIT 0.0001f

__hostdevice__ int cuda_isEqualReIndex(math::Matrix* matrix1,
                                       math::Matrix* matrix2, uintt index) {
  HOST_INIT();
  return fabs(matrix1->reValues[index] - matrix2->reValues[index]) <
                 COMPARE_LIMIT
             ? 1
             : 0;
}

__hostdevice__ int cuda_isEqualImIndex(math::Matrix* matrix1,
                                       math::Matrix* matrix2, uintt index) {
  HOST_INIT();
  return fabs(matrix1->imValues[index] - matrix2->imValues[index]) <
                 COMPARE_LIMIT
             ? 1
             : 0;
}

__hostdevice__ int cuda_isEqualRe(math::Matrix* matrix1, math::Matrix* matrix2,
                                  uintt column, uintt row) {
  HOST_INIT();
  return fabs(GetRe(matrix1, column, row) - GetRe(matrix2, column, row)) <
                 COMPARE_LIMIT
             ? 1
             : 0;
}

__hostdevice__ int cuda_isEqualIm(math::Matrix* matrix1, math::Matrix* matrix2,
                                  uintt column, uintt row) {
  HOST_INIT();
  return fabs(GetIm(matrix1, column, row) - GetIm(matrix2, column, row)) <
                 COMPARE_LIMIT
             ? 1
             : 0;
}

#endif  // CUCOMPAREUTILSCOMMON
