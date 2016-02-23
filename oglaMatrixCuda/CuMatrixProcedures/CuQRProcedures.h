/*
* File:   CuQRProcedures.h
* Author: mmatula
*
* Created on January 8, 2015, 9:26 PM
*/

#ifndef CUQRPROCEDURES_H
#define CUQRPROCEDURES_H

#include "CuCore.h"
#include "MatrixAPI.h"
#include "CuCopyProcedures.h"
#include "CuTransponseProcedures.h"
#include "CuDotProductProcedures.h"
#include "CuIdentityProcedures.h"
#include "CuMagnitudeUtils.h"
#include "CuMagnitudeVecOptProcedures.h"
#include "CuMultiplicationProcedures.h"
#include "CuSubstractionProcedures.h"

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
  Reset(G);

  if (tx == 0 && ty == 0) {
#if 0
      floatt reg = 0;
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
    floatt ic = 0;
#endif

    floatt s = 0;
    floatt is = 0;
    floatt c = 0;
    floatt ic = 0;
    if (NULL != A->reValues) {
      s = GetRe(A, column, row);
      c = GetRe(A, column, column);
    }
    if (NULL != A->imValues) {
      is = GetIm(A, column, row);
      ic = GetIm(A, column, column);
    }
    floatt r = sqrtf(c * c + s * s + is * is + ic * ic);
    c = c / r;
    ic = ic / r;
    s = s / r;
    is = is / r;
    if (NULL != G->reValues) {
      SetRe(G, column, row, -s);
      SetRe(G, column, column, c);
      SetRe(G, row, row, c);
      SetRe(G, row, column, s);
      CUDA_TEST_CODE(EXPECT_EQ(4, test::getSetValuesCountRe(G)););
    }
    if (NULL != G->imValues) {
      SetIm(G, column, row, -is);
      SetIm(G, column, column, ic);
      SetIm(G, row, row, ic);
      SetIm(G, row, column, is);
      CUDA_TEST_CODE(EXPECT_EQ(4, test::getSetValuesCountIm(G)););
    }
  }
  Reset(G);
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
  CUDA_copyMatrix(R1, A);
  uintt count = 0;
  for (uintt fa = 0; fa < A->columns; ++fa) {
    for (uintt fb = A->rows - 1; fb > fa; --fb) {
      floatt v = GetRe(A, fa, fb);
      if ((-0.0001 < v && v < 0.0001) == false) {
        CUDA_prepareGMatrix(R1, fa, fb, G, tx, ty);
        CUDA_dotProduct(R, G, R1, tx, ty);
        Push(R);
        if (count == 0) {
          CUDA_transposeMatrix(Q, G, tx, ty);
          Push(Q);
        } else {
          CUDA_transposeMatrix(GT, G, tx, ty);
          Push(GT);
          CUDA_dotProduct(Q, Q1, GT, tx, ty);
          Push(Q);
        }
        ++count;
        CUDA_switchPointer(&R1, &R);
        CUDA_switchPointer(&Q1, &Q);
      }
    }
  }
  if (count & 1 == 1) {
    CUDA_copyMatrix(rQ, Q1);
    Push(rQ);
    CUDA_copyMatrix(rR, R1);
    Push(rR);
  }
}

__hostdevice__ void CUDA_QRHT(math::Matrix* Q, math::Matrix* R, math::Matrix* A,
                              math::Matrix* AT, floatt* sum, floatt* buffer,
                              math::Matrix* P, math::Matrix* I, math::Matrix* v,
                              math::Matrix* vt, math::Matrix* vvt) {
  CUDA_TEST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;

  for (uintt column = 0; column < A->columns; ++column) {
    uintt row = column + 1;
    floatt a21 = A->reValues[A->columns * row + column];
    floatt sgna21 = 0;
    CUDA_getSgn(&sgna21, a21);
    CUDA_magnitudeOptVecEx(sum, A, column, row + 1, A->rows, buffer);
    floatt sum = 0;
    CUDA_getMagnitude(&sum, buffer);
    floatt alpha = -sgna21 * sum;
    floatt r = sqrtf(0.5f * (alpha * alpha - a21 * alpha));
    v->reValues[0] = 0;
    uintt rowy = threadIdx.y + blockDim.y * blockIdx.y;
    if (rowy < row) {
      v->reValues[v->columns * rowy] = 0;
    }
    if (rowy == row) {
      v->reValues[v->columns * rowy] = (a21 - alpha) / (2 * r);
    }
    if (rowy > row) {
      v->reValues[v->columns * rowy] =
          A->reValues[rowy * A->columns + column] / (2 * r);
    }
    CUDA_transposeMatrix(vt, v, tx, ty);
    CUDA_dotProduct(vvt, v, vt, tx, ty);
    CUDA_multiplyConstantMatrix(vvt, vvt, 2, 0, tx, ty);
    CUDA_substractMatrices(P, I, vvt, tx, ty);
    CUDA_dotProduct(Q, A, P, tx, ty);
    CUDA_dotProduct(R, P, Q, tx, ty);
  }
}

#endif /* CUQRPROCEDURES_H */
