/*
 * Copyright 2016 - 2021 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CUQRPROCEDURES_H
#define CUQRPROCEDURES_H

#include "CuCore.h"
#include "MatrixAPI.h"
#include "CuCopyProcedures.h"
#include "CuTransposeProcedures.h"
#include "CuDotProductSpecificProcedures.h"
#include "CuIdentityProcedures.h"
#include "CuMagnitudeUtils.h"
#include "CuMagnitudeVecOptProcedures.h"
#include "CuMultiplicationProcedures.h"
#include "CuSubtractionProcedures.h"
#include "CuSwitchPointer.h"

__hostdevice__ void CUDA_prepareGMatrix(math::Matrix* A, uintt column,
                                        uintt row, math::Matrix* G, uintt tx,
                                        uintt ty) {
  HOST_INIT();
  CUDA_SetIdentityMatrix(G);
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
    if (A->re.ptr != NULL) {
      s = GetRe(A, column, row);
      c = GetRe(A, column, column);
    }
    if (A->im.ptr != NULL) {
      is = GetIm(A, column, row);
      ic = GetIm(A, column, column);
    }
    floatt r = sqrtf(c * c + s * s + is * is + ic * ic);
    c = c / r;
    ic = ic / r;
    s = s / r;
    is = is / r;
    if (G->re.ptr != NULL) {
      SetRe(G, column, row, -s);
      SetRe(G, column, column, c);
      SetRe(G, row, row, c);
      SetRe(G, row, column, s);
    }
    if (G->im.ptr != NULL) {
      SetIm(G, column, row, -is);
      SetIm(G, column, column, ic);
      SetIm(G, row, row, ic);
      SetIm(G, row, column, is);
    }
  }
  threads_sync();
}

__hostdevice__ void CUDA_QRGR(math::Matrix* Q, math::Matrix* R, math::Matrix* A,
                              math::Matrix* Q1, math::Matrix* R1,
                              math::Matrix* G, math::Matrix* GT) {
  HOST_INIT();
  uintt tx = blockIdx.x * blockDim.x + threadIdx.x;
  uintt ty = blockIdx.y * blockDim.y + threadIdx.y;
  math::Matrix* rQ = Q;
  math::Matrix* rR = R;

  CUDA_copyMatrix(R1, A);

  uintt count = 0;

  const floatt tolerance = 0.00001;

  for (uintt fa = 0; fa < A->dim.columns; ++fa)
  {
    for (uintt fb = A->dim.rows - 1; fb > fa; --fb)
    {
      floatt v = GetRe(A, fa, fb);
      if ((-tolerance < v && v < tolerance) == false)
      {
        CUDA_prepareGMatrix(R1, fa, fb, G, tx, ty);
        CUDA_specific_dotProduct (R, G, R1);
        if (count == 0)
        {
          CUDA_transposeMatrix(Q, G);
        }
        else
        {
          CUDA_transposeMatrix(GT, G);
          CUDA_specific_dotProduct (Q, Q1, GT);
        }
        ++count;
        CUDA_switchPointer(&R1, &R);
        CUDA_switchPointer(&Q1, &Q);
      }
    }
  }
  if (Q1 == rQ)
  {
    CUDA_copyMatrix(rQ, Q1);
    CUDA_copyMatrix(rR, R1);
  }
}

#endif /* CUQRPROCEDURES_H */
