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

#ifndef OAP_CU_CALC_TRIANGULAR_MATRIX_H
#define OAP_CU_CALC_TRIANGULAR_MATRIX_H

#include "CuCore.h"
#include "Matrix.h"
#include "CuProcedures/CuQRProcedures_GR.h"
#include "CuProcedures/CuQRProcedures_HT.h"
#include "CuProcedures/CuIsUpperTriangularProcedures.h"

#define CUDA_calcUTMatrix_STEPS 10000

__hostdevice__ void CUDA_calcUTMatrix_GR(
    math::Matrix* H, math::Matrix* Q, math::Matrix* R,
    math::Matrix* aux1, math::Matrix* aux2, math::Matrix* aux3,
    math::Matrix* aux4, math::Matrix* aux5, math::Matrix* aux6)
{
  HOST_INIT();

  bool status = false;
  CUDA_SetIdentityMatrix(aux1);
  status = CUDA_isUpperTriangular(H);
  uint fa = 0;

  for (; fa < CUDA_calcUTMatrix_STEPS && status == false; ++fa)
  {
    CUDA_QRGR(Q, R, H, aux3, aux4, aux5, aux6);
    CUDA_specific_dotProduct(H, R, Q);
    CUDA_specific_dotProduct(aux2, Q, aux1);
    CUDA_switchPointer(&aux2, &aux1);
    status = CUDA_isUpperTriangular(H);
  }
  if (fa % 2 != 0)
  {
    CUDA_copyMatrix (Q, aux1);
  }
  else
  {
    CUDA_copyMatrix (Q, aux2);
  }
}

__hostdevice__ void CUDA_calcUTMatrixStep_GR(
    math::Matrix* H, math::Matrix* Q, math::Matrix* R,
    math::Matrix* aux1, math::Matrix* aux2, math::Matrix* aux3,
    math::Matrix* aux4, math::Matrix* aux5, math::Matrix* aux6)
{
  HOST_INIT();

  CUDA_SetIdentityMatrix(aux1);
  bool status = CUDA_isUpperTriangular(H);

  if (status == false)
  {
    CUDA_QRGR(Q, R, H, aux3, aux4, aux5, aux6);
    CUDA_specific_dotProduct(H, R, Q);
    CUDA_specific_dotProduct(aux2, Q, aux1);
  }

  CUDA_copyMatrix(Q, aux2);
}

__hostdevice__ void CUDAKernel_calcUTMatrix_HR(
    math::Matrix* H, math::Matrix* Q, math::Matrix* R,
    math::Matrix* aux1, math::Matrix* aux2, math::Matrix* V,
    math::Matrix* VT, math::Matrix* P, math::Matrix* VVT, floatt* sharedBuffer)
{
  HOST_INIT();

  bool status = false;
  CUDA_SetIdentityMatrix(aux1);
  status = CUDA_isUpperTriangular(H);
  uint fa = 0;

  for (; fa < CUDA_calcUTMatrix_STEPS && status == false; ++fa)
  {
    CUDA_QRHT (Q, R, H, V, VT, sharedBuffer, P, VVT);
    CUDA_specific_dotProduct(H, R, Q);
    CUDA_specific_dotProduct(aux2, Q, aux1);
    CUDA_switchPointer(&aux2, &aux1);
    status = CUDA_isUpperTriangular(H);
  }
  if (fa % 2 != 0)
  {
    CUDA_copyMatrix(Q, aux1);
  }
  else
  {
    CUDA_copyMatrix(Q, aux2);
  }
}

__hostdevice__ void CUDA_calcUTMatrixStep_HR(
    math::Matrix* H, math::Matrix* Q, math::Matrix* R,
    math::Matrix* aux1, math::Matrix* aux2, math::Matrix* V,
    math::Matrix* VT, math::Matrix* P, math::Matrix* VVT, floatt* sharedBuffer)
{
  HOST_INIT();

  bool status = false;
  CUDA_SetIdentityMatrix(aux1);
  uint fa = 0;
  status = CUDA_isUpperTriangular(H);

  for (; fa < CUDA_calcUTMatrix_STEPS && status == false; ++fa)
  {
    CUDA_QRHT (Q, R, H, V, VT, sharedBuffer, P, VVT);
    CUDA_specific_dotProduct(H, R, Q);
    CUDA_specific_dotProduct(aux2, Q, aux1);
    CUDA_switchPointer(&aux2, &aux1);
    status = CUDA_isUpperTriangular(H);
  }
  if (fa % 2 == 0)
  {
    CUDA_copyMatrix(Q, aux1);
  }
  else
  {
    CUDA_copyMatrix(Q, aux2);
  }
}

#endif  // CUTRIANGULARH
