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


#ifndef OAP_HOST_MATRIX_KERNELS_H
#define OAP_HOST_MATRIX_KERNELS_H

#include "Matrix.h"
#include "Logger.h"

#include <utility>
#include <math.h>
#include "CudaUtils.h"

namespace
{
inline void aux_swapPointers (math::ComplexMatrix** a, math::ComplexMatrix** b)
{
  math::ComplexMatrix* temp = *b;
  *b = *a;
  *a = temp;
}
}

template<typename CalcApi, typename MatrixApi>
void host_prepareGMatrix (math::ComplexMatrix* A, uintt column, uintt row, math::ComplexMatrix* G, CalcApi& capi, MatrixApi& mapi)
{
  capi.setIdentity (G);

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
  const auto minfo = mapi.getMatrixInfo (A);
  bool isre = minfo.isRe;
  bool isim = minfo.isIm;
  if (isre)
  {
    s = mapi.getReValue (A, column, row);
    c = mapi.getReValue (A, column, column);
  }
  if (isim)
  {
    is = mapi.getImValue (A, column, row);
    ic = mapi.getImValue (A, column, column);
  }

  floatt r = sqrt(c * c + s * s + is * is + ic * ic);
  c = c / r;
  ic = ic / r;
  s = s / r;
  is = is / r;

  if (isre)
  {
    mapi.setReValue (G, column, row, -s);
    mapi.setReValue (G, column, column, c);
    mapi.setReValue (G, row, row, c);
    mapi.setReValue (G, row, column, s);
  }

  if (isim)
  {
    mapi.setImValue (G, column, row, -is);
    mapi.setImValue (G, column, column, ic);
    mapi.setImValue (G, row, row, ic);
    mapi.setImValue (G, row, column, is);
  }
}

template<typename CalcApi, typename MatrixApi, typename CopyKernelMatrixToKernelMatrix>
bool HOSTKernel_QRGR (math::ComplexMatrix* Q, math::ComplexMatrix* R, math::ComplexMatrix* A,
                      math::ComplexMatrix* Q1, math::ComplexMatrix* R1, math::ComplexMatrix* G,
                      math::ComplexMatrix* GT, CalcApi& capi, MatrixApi& mapi,
                      CopyKernelMatrixToKernelMatrix&& copyKernelMatrixToKernelMatrix)
{
  math::ComplexMatrix* rQ = Q;
  math::ComplexMatrix* rR = R;

  copyKernelMatrixToKernelMatrix (R1, A);

  bool first = true;

  const auto minfo = mapi.getMatrixInfo (A);
  uintt Acolumns = minfo.columns();
  uintt Arows = minfo.rows();

  const floatt limit = 0.00001;

  for (uintt fa = 0; fa < Acolumns; ++fa)
  {
    for (uintt fb = Arows - 1; fb > fa; --fb)
    {
      floatt v =  mapi.getReValueIdx (A, fa + fb * Acolumns);

      if ((-limit < v && v < limit) == false)
      {
        host_prepareGMatrix(R1, fa, fb, G, capi, mapi);
        capi.dotProduct (R, G, R1);

        if (first)
        {
          capi.transpose(Q, G);
          first = false;
        }
        else
        {
          capi.transpose(GT, G);
          capi.dotProduct(Q, Q1, GT);
        }

        aux_swapPointers(&R1, &R);
        aux_swapPointers(&Q1, &Q);
      }
    }
  }
  if (Q1 != rQ)
  {
    copyKernelMatrixToKernelMatrix(rQ, Q1);
    copyKernelMatrixToKernelMatrix(rR, R1);
  }
  return true;
}


#endif  // HOSTMATRIXKERNELS_H
