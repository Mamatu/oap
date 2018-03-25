/*
 * Copyright 2016, 2017 Marcin Matula
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



#include "HostMatrixKernels.h"
#include <math.h>
#include "DeviceMatrixKernels.h"
#include "oapCudaMatrixUtils.h"

inline void aux_switchPointer(math::Matrix** a, math::Matrix** b) {
  math::Matrix* temp = *b;
  *b = *a;
  *a = temp;
}

inline CUresult host_prepareGMatrix(math::Matrix* A, uintt column, uintt row,
                                    math::Matrix* G, oap::cuda::Kernel& kernel) {
  CUresult result = DEVICEKernel_SetIdentity(G, kernel);
  if (result != CUDA_SUCCESS) {
    // return result;
  }

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
  uintt Acolumns = CudaUtils::GetColumns(A);
  bool isre = CudaUtils::GetReValues(A) != NULL;
  bool isim = CudaUtils::GetImValues(A) != NULL;
  if (isre) {
    s = CudaUtils::GetReValue(A, column + row * Acolumns);
    c = CudaUtils::GetReValue(A, column + column * Acolumns);
  }
  if (isim) {
    is = CudaUtils::GetImValue(A, column + row * Acolumns);
    ic = CudaUtils::GetImValue(A, column + column * Acolumns);
  }
  floatt r = sqrt(c * c + s * s + is * is + ic * ic);
  c = c / r;
  ic = ic / r;
  s = s / r;
  is = is / r;
  if (isre) {
    CudaUtils::SetReValue(G, column + row * Acolumns, -s);
    CudaUtils::SetReValue(G, column + (column)*Acolumns, c);
    CudaUtils::SetReValue(G, (row) + (row)*Acolumns, c);
    CudaUtils::SetReValue(G, (row) + (column)*Acolumns, s);
  }
  if (isim) {
    CudaUtils::SetImValue(G, column + row * Acolumns, -is);
    CudaUtils::SetImValue(G, column + (column)*Acolumns, ic);
    CudaUtils::SetImValue(G, (row) + (row)*Acolumns, ic);
    CudaUtils::SetImValue(G, (row) + (column)*Acolumns, is);
  }
  return CUDA_SUCCESS;
}

CUresult HOSTKernel_QRGR(math::Matrix* Q, math::Matrix* R, math::Matrix* A,
                         math::Matrix* Q1, math::Matrix* R1, math::Matrix* G,
                         math::Matrix* GT, oap::cuda::Kernel& kernel) {
  math::Matrix* rQ = Q;
  math::Matrix* rR = R;
  oap::cuda::CopyDeviceMatrixToDeviceMatrix(R1, A);
  uintt count = 0;
  uintt Acolumns = CudaUtils::GetColumns(A);
  uintt Arows = CudaUtils::GetRows(A);

  for (uintt fa = 0; fa < Acolumns; ++fa) {
    for (uintt fb = Arows - 1; fb > fa; --fb) {

      floatt v = CudaUtils::GetReValue(A, fa + fb * Acolumns);

      if ((-0.0001 < v && v < 0.0001) == false) {
        host_prepareGMatrix(R1, fa, fb, G, kernel);
        DEVICEKernel_DotProduct(R, G, R1, kernel);

        if (count == 0) {
          DEVICEKernel_Transpose(Q, G, kernel);
        } else {
          DEVICEKernel_Transpose(GT, G, kernel);
          DEVICEKernel_DotProduct(Q, Q1, GT, kernel);
        }

        ++count;
        aux_switchPointer(&R1, &R);
        aux_switchPointer(&Q1, &Q);
      }
    }
  }

  if (count % 2 != 0) {
    oap::cuda::CopyDeviceMatrixToDeviceMatrix(rQ, Q1);
    oap::cuda::CopyDeviceMatrixToDeviceMatrix(rR, R1);
  }
  return CUDA_SUCCESS;
}

void HOSTKernel_CalcTriangularH(math::Matrix* H1, math::Matrix* Q,
                                    math::Matrix* R1, math::Matrix* Q1,
                                    math::Matrix* QJ, math::Matrix* Q2,
                                    math::Matrix* R2, math::Matrix* G,
                                    math::Matrix* GT,
                                    oap::CuProceduresApi& cuMatrix, uint count) {
  bool status = false;
  cuMatrix.setIdentity(Q1);
  status = cuMatrix.isUpperTriangular(H1);
  for (uint idx = 0; idx < count && status == false; ++idx) {
    cuMatrix.QRGR(Q, R1, H1, Q2, R2, G, GT);
    cuMatrix.dotProduct(H1, R1, Q);
    cuMatrix.dotProduct(QJ, Q, Q1);
    aux_switchPointer(&QJ, &Q1);
    status = cuMatrix.isUpperTriangular(H1);
    // if (fb == 200) { abort();}
  }
  aux_switchPointer(&QJ, &Q1);
  oap::cuda::CopyDeviceMatrixToDeviceMatrix(Q, QJ);
}
