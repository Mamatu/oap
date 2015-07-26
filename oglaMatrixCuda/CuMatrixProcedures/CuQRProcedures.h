/*
 * File:   CuQRProcedures.h
 * Author: mmatula
 *
 * Created on January 8, 2015, 9:26 PM
 */

#ifndef CUQRPROCEDURES_H
#define CUQRPROCEDURES_H

#include "CuCore.h"
#include "CuCopyProcedures.h"
#include "CuTransponseProcedures.h"
#include "CuDotProductProcedures.h"
#include "CuIdentityProcedures.h"

__hostdevice__ void CUDA_switchPointer(math::Matrix** a, math::Matrix** b) {
  CUDA_TEST_INIT();
  math::Matrix* temp = *b;
  *b = *a;
  *a = temp;
  threads_sync();
}

__hostdevice__ void CUDA_prepareGMatrix(math::Matrix* A, uintt column,
                                        uintt row, math::Matrix* G, uintt tx,
                                        uintt ty) {
  CUDA_TEST_INIT();
  CUDA_SetIdentityMatrix(G, tx, ty);
  if (tx == 0 && ty == 0) {
    /*floatt reg = 0;
    floatt img = 0;
    floatt ref = 0;
    floatt imf = 0;
    if (A->reValues) {
      reg = A->reValues[column + row * A->columns];
      ref = A->reValues[column + column * A->columns];
    }
    if (A->imValues) {
      img = A->imValues[column + row * A->columns];
      imf = A->imValues[column + column * A->columns];
    }
    floatt r = sqrt(ref * ref + reg * reg + img * img + imf * imf);
    floatt lf = sqrt(ref * ref + imf * imf);
    floatt sign = 1;
    floatt isign = 0;
    if (fabs(ref) >= MATH_VALUE_LIMIT || fabs(imf) >= MATH_VALUE_LIMIT) {
      sign = ref / lf;
      isign = imf / lf;
    }
    floatt s = (sign * reg + img * isign) / r;
    floatt is = (isign * reg - img * sign) / r;
    floatt c = lf / r;
    floatt ic = 0;*/

    floatt s = 0;
    floatt is = 0;
    floatt c = 0;
    floatt ic = 0;
    if (NULL != A->reValues) {
      s = A->reValues[column + row * A->columns];
      c = A->reValues[column + column * A->columns];
    }
    if (NULL != A->imValues) {
      is = A->imValues[column + row * A->columns];
      ic = A->imValues[column + column * A->columns];
    }
    floatt r = sqrtf(c * c + s * s + is * is + ic * ic);
    c = c / r;
    ic = ic / r;
    s = s / r;
    is = is / r;
    if (NULL != G->reValues) {
      G->reValues[column + row * A->columns] = -s;
      G->reValues[column + (column)*A->columns] = c;
      G->reValues[(row) + (row)*A->columns] = c;
      G->reValues[(row) + (column)*A->columns] = s;
    }
    if (NULL != G->imValues) {
      G->imValues[column + row * A->columns] = -is;
      G->imValues[column + (column)*A->columns] = ic;
      G->imValues[(row) + (row)*A->columns] = ic;
      G->imValues[(row) + (column)*A->columns] = is;
    }
  }
  threads_sync();
}

__hostdevice__ void CUDA_QRGR(math::Matrix* Q, math::Matrix* R, math::Matrix* A,
                              math::Matrix* Q1, math::Matrix* R1,
                              math::Matrix* G, math::Matrix* GT) {
  CUDA_TEST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  math::Matrix* rQ = Q;
  math::Matrix* rR = R;
  CUDA_copyMatrix(R1, A, tx, ty);
  uintt count = 0;
  for (uintt fa = 0; fa < A->columns; ++fa) {
    for (uintt fb = A->rows - 1; fb > fa; --fb) {
      floatt v = A->reValues[fa + fb * A->columns];
      if ((-0.0001 < v && v < 0.0001) == false) {
        CUDA_prepareGMatrix(R1, fa, fb, G, tx, ty);
        CUDA_multiplyMatrices(R, G, R1, tx, ty);
        if (count == 0) {
          CUDA_transposeMatrix(Q, G, tx, ty);
        } else {
          CUDA_transposeMatrix(GT, G, tx, ty);
          CUDA_multiplyMatrices(Q, Q1, GT, tx, ty);
        }
        ++count;
        CUDA_switchPointer(&R1, &R);
        CUDA_switchPointer(&Q1, &Q);
      }
    }
  }
  if (count & 1 == 1) {
    CUDA_copyMatrix(rQ, Q1, tx, ty);
    CUDA_copyMatrix(rR, R1, tx, ty);
  }
}

__hostdevice__ void CUDA_QRHT() {
  CUDA_TEST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;

}

#endif /* CUQRPROCEDURES_H */
